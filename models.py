import torch
import torch.nn as nn
from torch import Tensor
from torchdiffeq import odeint
from data import (
    calculate_node_features,
    sample_prior,
    pad_graph_batches,
    get_dynamic_graph_batches,
    make_graph,
    position_to_velocity,
    velocity_to_acceleration,
    velocity_to_position,
    get_time_axis_idx,
)
from evaluation import (
    compute_similarities,
    only_evaluate,
    plot_evaluation,
    plot_trajectory_evolution,
    visualize_trajectories,
    calculate_intercollision_time,
    visualize_trajectories_paper,
)
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add, scatter_mean
from utils.unet import UNetModel, timestep_embedding
import numpy as np
from tqdm import tqdm


class DynamicGraphNetwork(nn.Module):
    class EGCL(nn.Module):
        def __init__(
            self,
            in_channels: int,
            in_edge_channels: int,
            hidden_channels: int,
            hidden_edge_channels: int,
            out_channels: int,
            layer_num: int,
            config: dict,
        ):
            """
            Graph Convolution layer that does spatial message passing with self-attention

            Args:
                in_channels: Number of input features per node
                in_edge_channels: Number of input features per edge
                hidden_channels: Number of hidden features per node
                hidden_edge_channels: Number of hidden features per edge
                out_channels: Number of output features per node
                layer_num: Layer number
                config: Configuration dictionary
            """
            super().__init__()

            self.in_channels = in_channels
            self.in_edge_channels = in_edge_channels
            self.hidden_channels = hidden_channels
            self.hidden_edge_channels = hidden_edge_channels
            self.out_channels = out_channels
            self.window = config["time_window"]
            self.layer_num = layer_num
            self.config = config

            # Node update network
            self.node_mlp = nn.Sequential(
                nn.Linear(
                    hidden_channels + hidden_edge_channels + hidden_channels // 4,
                    4 * hidden_channels,
                ),
                nn.SiLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(4 * hidden_channels, out_channels),
            )

            # Edge update network
            if (
                self.layer_num < len(self.config["model_layers"]) - 1
                and self.config["use_edge_mlp"]
            ):
                self.edge_mlp = nn.Sequential(
                    nn.Linear(
                        2 * hidden_channels + 1 + hidden_channels // 4,
                        4 * hidden_channels,
                    ),
                    nn.SiLU(),
                    nn.Dropout(config["dropout"]),
                    nn.Linear(4 * hidden_channels, hidden_edge_channels),
                )
            message_input_dim = (
                in_edge_channels + 2 * in_channels + 1 + 2 * (hidden_channels // 4)
            )

            # Graph message passing network
            self.message_mlp = nn.Sequential(
                nn.Linear(
                    message_input_dim,
                    4 * hidden_channels,
                ),
                nn.SiLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(4 * hidden_channels, hidden_edge_channels),
                nn.SiLU(),
            )

            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_channels // 4, 2 * hidden_channels),
                nn.SiLU(),
                nn.Linear(2 * hidden_channels, hidden_channels // 4),
            )

            if self.config["use_split_approach"]:
                self.pos_mlp = nn.Sequential(
                    nn.Linear(hidden_edge_channels, 4 * hidden_channels),
                    nn.SiLU(),
                    nn.Linear(4 * hidden_channels, config["dims"]),
                )

            # Layer normalization for stability
            self.layer_norm_nodes = nn.LayerNorm(hidden_channels)
            self.layer_norm_edges = nn.LayerNorm(hidden_edge_channels)

        def forward(
            self, x: Data, t: Tensor, pos: Tensor | None
        ) -> tuple[Tensor | None, Data]:
            """
            Forward pass of Equivariant Graph Convolutional Layer

            Args:
                x: Dynamic Graph with node and edge features
                pos: Positions of nodes [b*f*n, dims]
                t: Flow matching time  [bs]

            Returns:
                Updated vector field [b*f*n, dims]
                Updated node features in dynamic graph x
            """
            assert x.x is not None and x.edge_attr is not None and x.edge_index is not None

            # Get flow matching time embedding
            t_emb = timestep_embedding(t, self.hidden_channels // 4)
            t_emb_nodes = self.time_mlp(t_emb)[x.batch]
            t_emb_edges = t_emb_nodes[x.edge_index[0]]

            # Get frame number embedding
            frame_emb = x.frame_emb[x.edge_index[0]]

            # Collect messages from neighbors
            source_x, target_x = x.x[x.edge_index[0]], x.x[x.edge_index[1]]

            message_input = torch.cat(
                [source_x, target_x, x.edge_attr, x.dist, t_emb_edges, frame_emb], dim=-1
            )

            messages = self.message_mlp(message_input)

            if self.config["use_split_approach"] and pos is not None:
                pos_messages = self.pos_mlp(messages)
                pos_messages = pos_messages * (pos[x.edge_index[1]] - pos[x.edge_index[0]])

                # Aggregate positional messages
                aggr_pos_messages = scatter_mean(
                    pos_messages, x.edge_index[1], dim=0, dim_size=x.x.shape[0]
                )

                pos_v_t = aggr_pos_messages
            else:
                pos_v_t = None

            # Aggregate attended messages for each node update
            aggr_messages = scatter_add(
                messages, x.edge_index[1], dim=0, dim_size=x.x.shape[0]
            )

            # Update node features using time and aggregated messages
            node_update = torch.cat([x.x, aggr_messages, t_emb_nodes], dim=-1)

            node_update = self.node_mlp(node_update)

            x.x = self.layer_norm_nodes(node_update + x.x)

            if (
                self.layer_num < len(self.config["model_layers"]) - 1
                and self.config["use_edge_mlp"]
            ):
                edge_update = torch.cat(
                    [
                        x.x[x.edge_index[0]],
                        x.x[x.edge_index[1]],
                        x.dist,
                        t_emb_edges,
                    ],
                    dim=-1,
                )

                x.edge_attr = self.layer_norm_edges(self.edge_mlp(edge_update) + x.edge_attr)

            return pos_v_t, x

    class UNetLayer(nn.Module):
        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            layer_num: int,
            config: dict,
        ):
            """
            Temporal Convolution layer using UNet

            Args:
                in_channels: Number of input features per node
                hidden_channels: Number of hidden features per node
                out_channels: Number of output features per node
                layer_num: Layer number
                config: Configuration dictionary
            """
            super().__init__()

            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.window = config["time_window"]
            self.layer_num = layer_num
            self.config = config

            self.unet = UNetModel(
                image_size=self.window,
                dims=1,
                in_channels=in_channels,  # + self.time_channels,
                model_channels=hidden_channels,
                out_channels=hidden_channels,
                num_res_blocks=2,
                channel_mult=(1, 2),
                num_heads=2,
                num_head_channels=hidden_channels,
                attention_resolutions=[],
                use_scale_shift_norm=True,
                resblock_updown=True,
                dropout=config["dropout"],  # type: ignore
                cond_size=self.config["cond_frames"],
                use_new_attention_order=True,
            )

            self.vel_mlp = nn.Sequential(
                nn.Linear(hidden_channels, 4 * hidden_channels),
                nn.SiLU(),
                nn.Linear(4 * hidden_channels, config["dims"]),
            )

        def forward(self, x: Data, t: Tensor) -> tuple[Tensor | None, Tensor, Tensor]:
            """
            Apply temporal convolution to each node's feature sequence.

            Args:
                x: Node features tensor [num_nodes, hidden_channels]
                t: Flow matching time tensor [batch_size]

            Returns:
                Temporally convolved features [num_nodes, out_channels]
            """

            nodes_per_batch = scatter_add(torch.ones_like(x.batch), x.batch, dim=0)  # type: ignore

            indices = get_time_axis_idx(x.batch, self.config)  # type: ignore

            x_input = x.x[indices].permute(0, 2, 1)  # type: ignore

            # Reshape t to match batch * objects
            t = t.repeat_interleave(nodes_per_batch // self.window, dim=0)

            # Apply temporal convolution
            conv_output = self.unet(t, x_input)  # [batch * objects, out_channels, frames]

            conv_output = conv_output.permute(0, 2, 1).flatten(0, 1)

            hidden = torch.zeros_like(x.x)  # type: ignore
            idx = indices.flatten().unsqueeze(-1).expand(-1, hidden.size(-1))
            hidden = hidden.scatter(0, idx, conv_output)

            if self.config["use_split_approach"]:
                vel_change = self.vel_mlp(hidden)
            else:
                vel_change = None

            return vel_change, hidden, indices

    def __init__(
        self,
        in_channels: int,
        in_edge_channels: int,
        hidden_channels: int,
        hidden_edge_channels: int,
        out_channels: int,
        config: dict,
    ):
        super().__init__()

        self.config = config
        self.window = config["time_window"]
        self.dims = config["dims"]
        self.hidden_channels = hidden_channels

        node_input_dim = in_channels + hidden_channels // 4
        if "class" in config["node_features"]:
            node_input_dim += hidden_channels // 4

        self.node_embedding = nn.Sequential(
            nn.Linear(node_input_dim, 4 * hidden_channels),
            nn.SiLU(),
            nn.Linear(4 * hidden_channels, hidden_channels),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(in_edge_channels, 4 * hidden_channels),
            nn.SiLU(),
            nn.Linear(4 * hidden_channels, hidden_edge_channels),
        )
        if not self.config["use_split_approach"]:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels, 4 * hidden_channels),
                nn.SiLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(4 * hidden_channels, out_channels),
            )
            self.decoder2 = nn.Sequential(
                nn.Linear(hidden_edge_channels, 4 * hidden_channels),
                nn.SiLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(4 * hidden_channels, out_channels),
            )
            self.norm1 = nn.LayerNorm(hidden_channels)
            self.norm2 = nn.LayerNorm(hidden_channels)

        layers = []
        edge_embedders = []
        for i in range(len(config["model_layers"])):
            if config["model_layers"][i] == "UNet":
                layers.append(
                    self.UNetLayer(
                        in_channels=hidden_channels,
                        hidden_channels=hidden_channels,
                        out_channels=hidden_channels,
                        layer_num=i,
                        config=config,
                    )
                )
                edge_embedders.append(None)
            elif config["model_layers"][i] == "MP":
                layers.append(
                    self.EGCL(
                        in_channels=hidden_channels,
                        in_edge_channels=hidden_edge_channels,
                        hidden_channels=hidden_channels,
                        hidden_edge_channels=hidden_edge_channels,
                        out_channels=hidden_channels,
                        layer_num=i,
                        config=config,
                    )
                )
                edge_embedders.append(self.edge_embedding)
            elif config["model_layers"][i] == "ECMP":
                layers.append(
                    self.DynamicGraphEdgeConvLayer(
                        in_channels=hidden_channels,
                        in_edge_channels=hidden_edge_channels,
                        hidden_channels=hidden_channels,
                        hidden_edge_channels=hidden_edge_channels,
                        out_channels=hidden_channels,
                        layer_num=i,
                        config=config,
                    )
                )
            elif self.config["model_layers"][i] == "Fuse":
                layers.append(
                    self.SpatialTemporalFusion(
                        hidden_channels=hidden_channels,
                        config=config,
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.edge_embedders = nn.ModuleList(edge_embedders)

    def forward(self, x: Data, t: Tensor) -> Tensor:
        """Input shapes:
            x: Dynamic graph
            t: Flow matching time (batch_size)

        Output shape: (bs * num_nodes * num_frames, out_channels)
        """
        assert x.x is not None and x.batch is not None

        nodes_per_batch = scatter_add(torch.ones_like(x.batch), x.batch, dim=0)
        trajs_batch = nodes_per_batch // self.window

        frames = torch.arange(self.window, device=x.x.device, dtype=torch.int32)
        frame_emb = timestep_embedding(frames, self.hidden_channels // 4)

        pad_size, batch_size = trajs_batch.max(), x.num_graphs
        frame_idx = (
            torch.arange(self.window, device=x.x.device)
            .view(1, self.window, 1)
            .expand(batch_size, -1, pad_size)
        )
        pad_mask = torch.arange(pad_size, device=x.x.device).expand(
            batch_size, self.window, pad_size
        )
        pad_mask = pad_mask < trajs_batch.view(-1, 1, 1)
        frame_idx = frame_idx[pad_mask].flatten()
        frame_emb = frame_emb[frame_idx]
        x.frame_emb = frame_emb

        x.x = torch.cat(
            [x.x, frame_emb], dim=-1
        )  # Add frame embedding to node features per batch

        if "class" in self.config["node_features"]:
            class_idx = self.config["node_features"]["class"]
            class_emb = timestep_embedding(x.x[:, class_idx], self.hidden_channels // 4)
            x.x = torch.cat([x.x, class_emb], dim=-1)

        if self.config["predict_velocity"]:
            vel_idx = (
                [
                    self.config["node_features"]["vel_x"],
                    self.config["node_features"]["vel_y"],
                ]
                if self.config["dims"] == 2
                else [
                    self.config["node_features"]["vel_x"],
                    self.config["node_features"]["vel_y"],
                    self.config["node_features"]["vel_z"],
                ]
            )
            xt = x.x[:, vel_idx]
        elif not self.config["predict_velocity"] and not self.config["predict_acceleration"]:
            pos_idx = [
                self.config["node_features"]["x"],
                self.config["node_features"]["y"],
            ]
            xt = x.x[:, pos_idx]

        if self.config["use_inpainting_mask"] and x.mask is not None:
            x.x = torch.cat([x.x, x.mask], dim=-1)  # When inpainting give conditioning mask

        if not self.config["use_two_stage_approach"] and not self.config["use_split_approach"]:
            x.x = self.node_embedding(x.x)
            x.dist = x.edge_attr[:, :1]  # type: ignore
            x.edge_attr = self.edge_embedding(x.edge_attr)
            for layer in self.layers:
                if self.config["model_layers"][layer.layer_num] == "UNet":
                    _, x.x, time_axis_idx = layer(x=x, t=t)
                elif self.config["model_layers"][layer.layer_num] == "MP":
                    _, x = layer(x=x, t=t, pos=None)

            x.time_axis_idx = time_axis_idx
            x.x = self.norm1(x.x)
            output = self.decoder(x.x)

        elif self.config["use_two_stage_approach"]:
            x.x = self.node_embedding(x.x)
            _, x.x, time_axis_idx = self.layers[0](x=x, t=t)  # UNet

            x.time_axis_idx = time_axis_idx
            unet_output = self.decoder(x.x)

            x.unet_output = unet_output

            x1_trajs = x.trajectories[time_axis_idx]
            if self.config["predict_velocity"]:
                pred_vels = xt + unet_output
                flat_pos = velocity_to_position(
                    x1_trajs, pred_vels, time_axis_idx, self.config
                )
                flat_vels = pred_vels
            elif (
                not self.config["predict_velocity"] and not self.config["predict_acceleration"]
            ):
                pred_pos = xt + unet_output
                flat_vels = position_to_velocity(
                    pred_pos, x.trajectories, time_axis_idx, self.config
                )
                flat_pos = pred_pos

            e_indexes, e_attrs = make_graph(
                flat_pos,
                flat_vels,
                self.config,
                frame_batch=frame_idx,
                batch=x.batch,  # type: ignore
                context=x.context,  # type: ignore
            )
            x.edge_index = e_indexes  # type: ignore
            x.edge_attr = e_attrs  # type: ignore
            x.dist = x.edge_attr[:, :1]

            x.edge_attr = self.edge_embedding(x.edge_attr)

            residuals = None
            for i, layer in enumerate(self.layers):
                if i == 0:
                    residuals = x.x
                    continue
                elif self.config["model_layers"][layer.layer_num] == "UNet":
                    _, x.x, time_axis_idx = layer(x=x, t=t)
                else:
                    _, x = layer(x=x, t=t, pos=None)
                if i < len(self.layers) - 1:
                    residuals = residuals + x.x  # type: ignore

            x.x = self.norm1(x.x + residuals)  # type: ignore

            x.interaction_output = self.decoder2(x.x)

            output = x.interaction_output  # + unet_output

        elif self.config["use_split_approach"]:
            xt, vels = x.x[:, vel_idx], x.x[:, vel_idx]
            x.time_axis_idx = get_time_axis_idx(x.batch, self.config)
            pos = velocity_to_position(x.trajectories, vels, x.time_axis_idx, self.config)

            x.x = self.node_embedding(x.x)
            x.edge_attr = self.edge_embedding(x.edge_attr)

            for i in range(0, len(self.layers)):
                if self.config["model_layers"][i] == "UNet":
                    v_t, x.x, x.time_axis_idx = self.layers[i](x=x, t=t)  # UNet
                    vels = vels + v_t

                elif self.config["model_layers"][i] == "MP":
                    pos = velocity_to_position(
                        x.trajectories, vels, x.time_axis_idx, self.config
                    )

                    if i > 1:
                        accs = velocity_to_acceleration(
                            vels, x.x1, x.time_axis_idx, self.config
                        )
                        node_feats = calculate_node_features(
                            pos, vels, accs, self.config, x.time_axis_idx
                        )
                        e_indexes, e_attrs = make_graph(
                            pos,
                            node_feats,
                            self.config,
                            frame_batch=frame_idx,
                            batch=x.batch,  # type: ignore
                            context=x.context,
                        )
                        x.edge_index, x.edge_attr = e_indexes, e_attrs  # type: ignore
                        x.dist = x.edge_attr[:, :1]
                        x.edge_attr = self.edge_embedders[i](x.edge_attr)

                    pos_v_t, x = self.layers[i](x=x, t=t, pos=pos)  # Message passing
                    pos = pos + pos_v_t

                    vels = position_to_velocity(
                        pos, x.trajectories, x.time_axis_idx, self.config
                    )

            output = vels - xt

        if self.config["use_inpainting_mask"] and x.mask is not None:
            output = output * x.mask

        return output  # (bs * num_nodes * num_frames, out_channels)

    def sample_flow(self, cfm, data: Data) -> tuple:
        """Sample t, xt and ut using conditional flow matching."""
        assert data.batch is not None

        # Pad batches to equal amount of trajectories in each batch
        x1, x1_pos = pad_graph_batches(
            data.x1,
            data.trajectories,
            data.batch,
            self.config,
            self.config["batch_size"],
        )

        # Sample padded prior for FM (and inpainting masks if used)
        x0, x0_pos, masks = sample_prior(x1_pos, data.batch, self.config, inference=False)  # type: ignore

        x1, x1_pos = x1.flatten(1, 2), x1_pos.flatten(1, 2)
        t_exp = self.config["t_sample_dist"]
        t = torch.rand(x1.shape[0], device=x0.device, dtype=x0.dtype) ** t_exp
        if self.config["use_dynamic_edges"] and self.config["batch_size"] == 1:
            t = t[0].repeat(x1.shape[0])  # If graph is used, we cannot vary t for each node
        if not self.config["predict_velocity"] and not self.config["predict_acceleration"]:
            x1_pos = x1

        # Sample xt on gaussian probability path using OT and corresponding vector field ut
        if self.config["use_optimal_transport"]:
            y = torch.arange(0, x0.shape[0], device=x1.device)
            t, xt, ut, x0_idx, x1_idx = cfm.guided_sample_location_and_conditional_flow(
                x0, x1, t=t, y0=y, y1=y
            )
        else:
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1, t=t)
            x0_idx, x1_idx = None, None

        xt_graph, xt, ut = get_dynamic_graph_batches(
            xt, ut, data, x0_pos, x1_pos, self.config, masks, x0_idx, x1_idx
        )

        return t, xt_graph, ut, xt

    def generate_batch(
        self, cfm, batch_graph: Data, steps: int = 20
    ) -> tuple[list, Tensor, list, list]:
        """Generate trajectories for one batch of graphs using the trained vector field.
        Returns: Tensor of shape (batch, num_nodes, num_frames, dims)"""
        assert batch_graph.batch is not None
        inference_graph = batch_graph.clone()  # type: ignore
        n_batches = batch_graph.num_graphs
        x1, x1_trajs = pad_graph_batches(
            inference_graph.x1,
            inference_graph.trajectories,
            batch_graph.batch,
            self.config,
            n_batches,
        )
        x0, x0_pos, masks = sample_prior(
            x1_trajs, batch_graph.batch, self.config, inference=True
        )  # type: ignore
        x0 = x0.view(n_batches, self.config["time_window"], -1, self.config["dims"])
        if not self.config["predict_velocity"] and not self.config["predict_acceleration"]:
            x1_trajs = x1

        def vector_field(t, x):
            t = t.repeat(n_batches)

            graph, _, _ = get_dynamic_graph_batches(
                x.flatten(1, 2),
                torch.zeros_like(x.flatten(1, 2)),
                inference_graph.clone(),
                x0_pos,
                x1_trajs,
                self.config,
                masks,
                None,
                None,
            )

            output = cfm.forward(t, graph)

            output, _ = pad_graph_batches(
                output,
                x1_trajs.flatten(0, 2),
                graph.batch,  # type: ignore
                self.config,
                n_batches,
            )

            return output

        t = torch.linspace(start=0, end=1, steps=steps, device=x0.device)
        t = t ** self.config["t_sample_dist"]
        solver = self.config["adaptive_solver"]
        if solver is None:
            solver = "euler"
        predictions_over_time = odeint(
            func=vector_field,  # type: ignore
            y0=x0,
            t=t,
            method=solver,
            atol=1e-3,
            rtol=1e-3,
            # options={"max_num_steps": 101} if solver == "dopri5" else None,
        )
        assert isinstance(predictions_over_time, torch.Tensor)

        predictions = predictions_over_time[-1]

        start_pos = x1_trajs[:, 0].unsqueeze(1)
        pred = predictions.cpu()
        all_trajs = predictions_over_time
        if self.config["predict_acceleration"]:
            start_vel = x1_trajs[:, 1].unsqueeze(1) - start_pos
            vels = torch.cat([start_vel, predictions], dim=1).cumsum(dim=1)[:, :-1]
            x0_vels = torch.cat([start_vel, x0], dim=1).cumsum(dim=1)[:, :-1]
            rollout_traj = torch.cat((start_pos, vels), dim=1).cumsum(dim=1)[:, :-1]
            priors_pos = torch.cat((start_pos, x0_vels), dim=1).cumsum(dim=1)[:, :-1]
        elif self.config["predict_velocity"]:
            rollout_traj = torch.cat((start_pos, predictions), dim=1).cumsum(dim=1)[:, :-1]
            priors_pos = torch.cat((start_pos, x0), dim=1).cumsum(dim=1)[:, :-1]
        else:
            # Revert min-max scaling
            rollout_traj = predictions_over_time[-1]
            priors_pos = x0

        # Integrate velocity for all time steps
        s_pos_all = start_pos.unsqueeze(0).repeat(steps, 1, 1, 1, 1)
        if self.config["predict_acceleration"]:
            s_vel = start_vel.unsqueeze(0).repeat(steps, 1, 1, 1, 1)
            vels = torch.cat((s_vel, predictions_over_time), dim=2).cumsum(dim=2)[:, :, :-1]
            all_trajs = torch.cat((s_pos_all, vels), dim=2).cumsum(dim=2)[:, :, :-1]
        elif self.config["predict_velocity"]:
            vels = predictions_over_time
            all_trajs = torch.cat((s_pos_all, vels), dim=2).cumsum(dim=2)[:, :, :-1]
        else:
            all_trajs = predictions_over_time

        rollouts, all_trajs_list, priors = [], [], []  # Unpad batches and add to lists
        node_counts = torch.bincount(batch_graph.batch, minlength=batch_graph.num_graphs)
        traj_counts = node_counts // self.config["time_window"]
        for i, rollout in enumerate(rollout_traj):
            rollouts.append(rollout[:, : traj_counts[i]].transpose(0, 1).cpu())
            all_trajs_list.append(all_trajs[:, i, :, : traj_counts[i]].transpose(1, 2).cpu())
            priors.append(priors_pos[i, :, : traj_counts[i]].transpose(0, 1).cpu())

        return rollouts, pred, all_trajs_list, priors  # (trajectories, num_frames, dims)

    def evaluate(
        self,
        graph: Data,
        priors: list[Tensor],
        trajectories: list[Tensor],
        testing: bool,
        debugging: bool,
    ) -> tuple:
        if not hasattr(graph, "batch") or not hasattr(graph, "num_graphs"):
            graph = Batch.from_data_list([graph.clone()])  # type: ignore
        plots = {}
        if self.config["use_best_of_20"] and testing:
            idxs = graph.num_graphs * 20
        elif self.config["use_mean_of_5"] and testing:
            idxs = graph.num_graphs * 5
        else:
            idxs = graph.num_graphs
        x1s = []
        for i in tqdm(range(idxs), desc="Evaluating Results"):
            if self.config["use_best_of_20"] and testing:
                batch_graph = graph.get_example(i // 20)
            elif self.config["use_mean_of_5"] and testing:
                batch_graph = graph.get_example(i // 5)
            else:
                batch_graph = graph.get_example(i)
            x0 = priors[i]  # (#trajs, #frames, dims)
            x1 = batch_graph.trajectories

            x1 = x1.view(self.window, -1, self.config["dims"]).transpose(0, 1)
            x1s.append(x1)

            if i == 0:
                aggregated_metrics = only_evaluate(x0, trajectories[i], x1, self.config)
                similarities = compute_similarities(trajectories[i], x1, self.config)
                for key, value in similarities.items():
                    aggregated_metrics[key] = [float(value)]

                traj_evolution = plot_trajectory_evolution(
                    x0.cpu(),
                    trajectories[i],
                    x1.cpu(),
                    fig_size=(18, 6),
                    max_trajectories=50,
                    max_x_and_y=self.config["max_x_and_y"],
                )
                plots["trajectory_evolution"] = traj_evolution
            else:
                new_metrics = only_evaluate(x0, trajectories[i], x1, self.config)
                similarities = compute_similarities(trajectories[i], x1, self.config)
                new_metrics.update(similarities)
                for key, value in new_metrics.items():
                    if type(value) is float:
                        aggregated_metrics[key].append(float(value))
                    elif type(value) is list:
                        for i, data in enumerate(value):
                            data = np.array(data)
                            aggregated_metrics[key][i] = np.concatenate(  # type: ignore
                                (aggregated_metrics[key][i], data)
                            )

        if self.config["use_best_of_20"] and testing:
            splits = graph.num_graphs
            ade_splits = np.array_split(aggregated_metrics["ade"], splits)  # type: ignore
            min_ade_values = np.array([np.min(ade_split) for ade_split in ade_splits])
            fde_splits = np.array_split(aggregated_metrics["fde"], splits)  # type: ignore
            min_fde_values = np.array([np.min(fde_split) for fde_split in fde_splits])
            aggregated_metrics["best_of_20_ade"] = float(np.mean(min_ade_values))
            aggregated_metrics["best_of_20_fde"] = float(np.mean(min_fde_values))
        elif self.config["use_mean_of_5"] and testing:
            splits = graph.num_graphs
            ade_splits = np.array_split(aggregated_metrics["ade"], splits)  # type: ignore
            ade_values = np.array([np.mean(ade_split) for ade_split in ade_splits])
            ade_std = np.array([np.std(ade_split) for ade_split in ade_splits])
            fde_splits = np.array_split(aggregated_metrics["fde"], splits)  # type: ignore
            fde_values = np.array([np.mean(fde_split) for fde_split in fde_splits])
            fde_std = np.array([np.std(fde_split) for fde_split in fde_splits])
            aggregated_metrics["mean_of_5_ade"] = float(np.mean(ade_values))
            aggregated_metrics["mean_of_5_fde"] = float(np.mean(fde_values))
            aggregated_metrics["mean_of_5_ade_std"] = float(np.mean(ade_std))
            aggregated_metrics["mean_of_5_fde_std"] = float(np.mean(fde_std))

        # Calculate Intercollision time
        bins, times, n_coll = calculate_intercollision_time(trajectories, self.config)
        _, times_gt, n_coll_gt = calculate_intercollision_time(x1s, self.config)
        aggregated_metrics["intercollision_time"] = [times, times_gt]
        aggregated_metrics["num_collisions"] = [n_coll, n_coll_gt]
        aggregated_metrics["pedestrian_density"] = bins

        plots.update(plot_evaluation(aggregated_metrics, self.config, testing))

        # Remove all lists from aggregated metrics and calculate mean
        keys = list(aggregated_metrics.keys())
        for key in keys:
            if (
                type(aggregated_metrics[key]) is list
                and type(aggregated_metrics[key][0]) is float
            ):
                aggregated_metrics[key] = np.mean(aggregated_metrics[key])

            elif type(aggregated_metrics[key]) is not float:
                aggregated_metrics.pop(key, None)

        if testing:
            if self.config["use_best_of_20"]:
                trajs = [trajectories[i * 20] for i in range(graph.num_graphs)]
            elif self.config["use_mean_of_5"]:
                trajs = [trajectories[i * 5] for i in range(graph.num_graphs)]
            else:
                trajs = trajectories
            if self.config["dims"] == 2:
                plots["rollouts"] = visualize_trajectories(
                    graph.trajectories.cpu(),  # type: ignore
                    trajs,
                    graph.batch,
                    num_samples=4,
                    history_frames=self.config["cond_frames"],
                    prediction_frames=self.config["time_window"] - self.config["cond_frames"],
                )
                visualize_trajectories_paper(
                    graph.trajectories.cpu(),  # type: ignore
                    trajs,
                    graph.batch,
                    num_samples=3,
                    history_frames=self.config["cond_frames"],
                    prediction_frames=self.config["time_window"] - self.config["cond_frames"],
                )

        return plots, aggregated_metrics

    def compute_loss(self, t: Tensor, vt: Tensor, ut: Tensor, xt_graph: Data, xt: Tensor):
        loss = 0
        mask = xt_graph.mask
        losses = self.config["loss_functions"]

        squared_error = (vt - ut) ** 2
        masked_error = squared_error * mask
        loss += torch.mean(masked_error) * self.config["loss_functions"]["mse"]

        if "relative_position" in losses and losses["relative_position"] > 0:
            time_axis_idx = xt_graph.time_axis_idx
            x1_pos = xt_graph.trajectories[time_axis_idx]

            interaction_correction = xt_graph.interaction_output

            if self.config["predict_velocity"]:
                x_vel = xt[time_axis_idx] + xt_graph.unet_output[time_axis_idx]
                x_pos = torch.cat([x1_pos[:, :1], x_vel], dim=1).cumsum(dim=1)[:, :-1]
            else:
                return ValueError("Not implemented yet")

            flat_pos = torch.zeros_like(xt_graph.x1)
            flat_pos[time_axis_idx.flatten(), :] = x_pos.flatten(0, 1)

            x1_correction = xt_graph.trajectories - flat_pos

            errors = torch.norm((x1_correction - interaction_correction), dim=-1)

            interaction_loss = torch.mean(errors)
            loss += interaction_loss * self.config["loss_functions"]["relative_position"]

        if "collision" in losses and losses["collision"] > 0:
            time_axis_idx = xt_graph.time_axis_idx
            x1_pos = xt_graph.trajectories[time_axis_idx]

            if self.config["predict_velocity"]:
                x_vel = xt[time_axis_idx] + vt[time_axis_idx]
                x_pos = torch.cat([x1_pos[:, :1], x_vel], dim=1).cumsum(dim=1)[:, :-1]
            else:
                return ValueError("Not implemented yet")

            flat_pos = torch.zeros_like(xt_graph.x1)
            flat_pos[time_axis_idx.flatten(), :] = x_pos.flatten(0, 1)

            x_rel_pos = flat_pos[xt_graph.edge_index[0]] - flat_pos[xt_graph.edge_index[1]]  # type: ignore
            x_distances = torch.norm(x_rel_pos, dim=-1)
            x_distances = torch.where(
                x_distances < self.config["repulsion_radius"],
                x_distances,
                torch.zeros_like(x_distances),
            )
            x1_rel_pos = (
                xt_graph.trajectories[xt_graph.edge_index[0]]  # type: ignore
                - xt_graph.trajectories[xt_graph.edge_index[1]]  # type: ignore
            )
            x1_distances = torch.norm(x1_rel_pos, dim=-1)
            x1_distances = torch.where(
                x1_distances < self.config["repulsion_radius"],
                x1_distances,
                torch.zeros_like(x1_distances),
            )

            collisions = torch.where(
                x_distances < x1_distances,
                self.config["repulsion_radius"] - x_distances,
                torch.zeros_like(x_distances),
            )

            penalties = torch.exp(collisions * 10) - 1

            collision_loss = torch.mean(penalties)
            loss += collision_loss * self.config["loss_functions"]["collision"]

        return loss
