import random
import lightning as L
from torch.utils.data import Dataset, random_split
from preprocess import (
    preprocess_pedestrians,
    calculate_edge_features,
)
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric import loader
from torch_geometric.nn import knn_graph, radius_graph
from torch_scatter import scatter_add
from tqdm import tqdm
import os


def sample_prior(
    trajectories: Tensor, batch: Tensor, config: dict, inference: bool
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Samples prior x0 in padded batches based the given config
    Args: trajectories: Tensor of shape (batch, num_frames, max_num_nodes, 2)
    Returns: Tensors of shape (batch, num_frames * max_num_nodes, 2)"""

    if config["use_inpainting_mask"]:
        window = 0
    else:
        window = config["cond_frames"]

    if trajectories.dim() == 3:
        trajectories = trajectories.unsqueeze(0)

    n_frames, n_batches = trajectories.shape[1], trajectories.shape[0]
    trajs = trajectories
    vels = torch.diff(trajs, dim=1)
    vels = torch.cat([vels, vels[:, -1:]], dim=1)

    if config["use_inpainting_mask"]:
        if config["use_uniform_mask"]:
            mask_length = (torch.rand_like(vels[:, 0, :, 0]) * config["cond_frames"]).int()
        else:
            mask_length = torch.tensor(config["cond_frames"] - 1, device=vels.device).repeat(
                vels.shape[0], vels.shape[2]
            )
        mask = torch.ones_like(trajs[:, :, :, :1]).to(dtype=torch.long)
        indices = torch.arange(n_frames, device=mask.device).reshape(1, -1, 1)
        mask_length = mask_length.unsqueeze(1)
        mask[indices < mask_length] = 0

    if config["prior_type"] != "position_gaussian" and config["prior_type"] != "full_gaussian":
        if config["prior_type"] == "gaussian":
            priors = sample_gaussian_prior(
                trajs[:, window - 1],
                n_frames - window,
                config["prior_gauss_std"],
                config["predict_velocity"],
                config,
            )
        elif (
            config["prior_type"] == "informed_walk"
            or config["prior_type"] == "informed_full_covariance_walk"
            or inference
        ):
            if not config["use_inpainting_mask"]:
                mask = torch.ones_like(vels[:, :, :, :1]).to(dtype=torch.long)
                mask[:, :window] = 0
            priors = sample_informed_walk_prior(vels, config, inference)
        else:
            ValueError("Other priors not yet implemented")

        if (config["predict_velocity"] or config["predict_acceleration"]) and config[
            "use_inpainting_mask"
        ]:
            priors[indices < mask_length] = vels[indices < mask_length]  # type: ignore
            priors_pos = torch.cat([trajs[:, :1], priors], dim=1)
            priors_pos = torch.cumsum(priors_pos, dim=1)[:, :-1]
            priors = torch.diff(priors_pos, dim=1)
            priors = torch.cat([priors, priors[:, -1:]], dim=1)
        elif (
            not config["predict_velocity"]
            and not config["predict_acceleration"]
            and config["use_inpainting_mask"]
        ):
            priors[indices < mask_length] = vels[indices < mask_length]  # type: ignore
            priors_pos = torch.cat([trajs[:, :1], priors], dim=1)
            priors_pos = torch.cumsum(priors_pos, dim=1)[:, :-1]
            priors = priors_pos
        elif not config["use_inpainting_mask"] and (
            config["predict_velocity"] or config["predict_acceleration"]
        ):
            priors = torch.cat([vels[:, :window], priors], dim=1)
            priors_pos = torch.cat([trajs[:, :1], priors], dim=1)
            priors_pos = torch.cumsum(priors_pos, dim=1)[:, :-1]
            priors = torch.diff(priors_pos, dim=1)
            priors = torch.cat([priors, priors[:, -1:]], dim=1)
        else:
            priors_pos = torch.cat([trajs[:, :window], priors], dim=1)
            priors = priors_pos

        if config["predict_acceleration"]:
            priors = torch.diff(priors, dim=1)
            priors = torch.cat([priors, priors[:, -1:]], dim=1)
        elif not config["predict_velocity"] and not config["predict_acceleration"]:
            # x_0, y_0 = priors[..., :1], priors[..., :1]
            # x_0 = (x_0 - config["x_range"][0]) / (config["x_range"][1] - config["x_range"][0])
            # y_0 = (y_0 - config["y_range"][0]) / (config["y_range"][1] - config["y_range"][0])
            # priors = torch.cat([x_0, y_0], dim=-1)
            priors_pos = priors
    elif config["prior_type"] == "position_gaussian":
        x_c = (
            trajs[:, config["cond_frames"] - 1].unsqueeze(1).expand(-1, trajs.shape[1], -1, -1)
        )
        priors_pos = x_c + torch.randn_like(trajs) * (config["prior_gauss_std"] ** 0.5)
        priors_pos[:, : config["cond_frames"]] = trajs[:, : config["cond_frames"]]
        priors = torch.diff(priors_pos, dim=1)
        priors = torch.cat([priors, priors[:, -1:]], dim=1)
    elif config["prior_type"] == "full_gaussian":
        priors_pos = torch.randn_like(trajs) * (config["prior_gauss_std"] ** 0.5)
        priors_pos[:, : config["cond_frames"]] = trajs[:, : config["cond_frames"]]
        priors = torch.diff(priors_pos, dim=1)
        priors = torch.cat([priors, priors[:, -1:]], dim=1)

    if config["give_last_frame"]:
        if config["predict_velocity"]:
            priors[:, -1] = vels[:, -1]
            mask[:, -1] = mask[:, 0]

    priors, priors_pos = priors.flatten(1, 2), priors_pos.flatten(1, 2)
    mask = mask.flatten(1, 2)

    return priors, priors_pos, mask


def sample_gaussian_prior(
    initial: Tensor, length: int, std: float, vel: bool, config: dict
) -> Tensor:
    """Sample priors from a gaussian with given standard deviation"""
    batch_size, num_traj, dim = initial.shape
    prior_normal = torch.randn(
        batch_size,
        length,
        num_traj,
        dim,
        device=initial.device,
        dtype=torch.float16 if config["use_fp16"] else initial.dtype,
    )
    if vel:
        prior = prior_normal * std
    else:
        prior = prior_normal * std + initial.unsqueeze(1)
        prior[:, 0] = initial
    return prior


def sample_informed_walk_prior(
    gt_velocities: Tensor, config: dict, inference: bool = False
) -> Tensor:
    """Sample prior trajectories based on given velocities, only using information in mask
    Args: initial_velocities: Tensor of shape (batch, num_frames, num_nodes, dims)
    mask: Tensor of shape (batch, num_frames, num_nodes, dims)
    Returns: Tensor of shape (batch, num_frames, num_nodes, dims)
             Tensor of shape (batch)"""
    n_batch, num_frames, num_traj, dim = gt_velocities.shape

    # Get mean and std of initial velocities per batch per trajectory using mask
    velocities = gt_velocities[:, : config["cond_frames"] - 1]
    if config["prior_type"] == "informed_walk":
        means = torch.mean(velocities, dim=1)
        means = means.unsqueeze(1).expand(-1, num_frames, -1, -1)
        std = torch.std(velocities, dim=1)
        std = std.unsqueeze(1).expand(-1, num_frames, -1, -1)

        # Increase variance of prior
        if inference:
            std = std * config["inference_prior_noise_factor"]
        else:
            std = std * config["prior_noise_factor"]

        # Sample from normal distribution with mean and std of conditional velocities
        prior = torch.normal(means, std).to(gt_velocities.device)

    elif config["prior_type"] == "informed_full_covariance_walk":
        velocities = velocities.permute(0, 2, 1, 3)  # (B, N, F, D)
        means = torch.mean(velocities, dim=2, keepdim=True)
        residuals = velocities - means
        covariances = torch.einsum("bnfd,bnfe->bnde", residuals, residuals) / (num_frames - 1)

        prior = torch.distributions.MultivariateNormal(means, covariances).sample()

    return prior


def augment_by_rotation_2d(graph: Data, rotation: Tensor, config: dict) -> Data:
    """Augment the given graph by rotation"""
    assert graph.x is not None
    rot_matrix = torch.tensor(
        [
            [torch.cos(rotation), -torch.sin(rotation)],
            [torch.sin(rotation), torch.cos(rotation)],
        ],
        dtype=graph.x.dtype,
        device=graph.x.device,
    )

    augmented_graph = graph.clone()
    assert augmented_graph.x is not None
    pos_id = (0, 1)

    augmented_graph.x[:, pos_id] = torch.matmul(graph.x[:, pos_id], rot_matrix.T)
    if "x" in config["node_features"] and "y" in config["node_features"]:
        i, j = config["node_features"]["x"], config["node_features"]["y"]
        augmented_graph.x[:, (i, j)] = torch.matmul(graph.x[:, (i, j)], rot_matrix.T)
    if (
        "acceleration_x" in config["node_features"]
        and "acceleration_y" in config["node_features"]
    ):
        i, j = (
            config["node_features"]["acceleration_x"],
            config["node_features"]["acceleration_y"],
        )
        augmented_graph.x[:, (i, j)] = torch.matmul(graph.x[:, (i, j)], rot_matrix.T)
    augmented_graph.priors[:, pos_id] = torch.matmul(graph.priors[:, pos_id], rot_matrix.T)
    augmented_graph.x1[:, pos_id] = torch.matmul(graph.x1[:, pos_id], rot_matrix.T)

    augmented_graph.priors_pos[:, (0, 1)] = torch.matmul(
        graph.priors_pos[:, (0, 1)], rot_matrix.T
    )
    augmented_graph.trajectories = torch.matmul(graph.trajectories, rot_matrix.T)

    if graph.edge_index is not None and graph.edge_attr is not None:
        source, target = graph.edge_index
        augmented_graph.edge_attr[:, config["edge_features"]["delta_x"]] = (  # type: ignore
            augmented_graph.x[source, pos_id[0]] - augmented_graph.x[target, pos_id[0]]
        )
        augmented_graph.edge_attr[:, config["edge_features"]["delta_y"]] = (  # type: ignore
            augmented_graph.x[source, pos_id[1]] - augmented_graph.x[target, pos_id[1]]
        )
    assert graph.x.shape[0] == augmented_graph.x.shape[0]

    return augmented_graph


def augment_by_rotation_3d(graph: Data, rotation: Tensor, config: dict) -> Data:
    """Augment the given graph by rotation"""
    assert graph.x is not None
    rot_matrix_x = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(rotation[0]), -torch.sin(rotation[0])],
            [0, torch.sin(rotation[0]), torch.cos(rotation[0])],
        ],
        dtype=graph.x.dtype,
        device=graph.x.device,
    )
    rot_matrix_y = torch.tensor(
        [
            [torch.cos(rotation[1]), 0, torch.sin(rotation[1])],
            [0, 1, 0],
            [-torch.sin(rotation[1]), 0, torch.cos(rotation[1])],
        ],
        dtype=graph.x.dtype,
        device=graph.x.device,
    )
    rot_matrix_z = torch.tensor(
        [
            [torch.cos(rotation[2]), -torch.sin(rotation[2]), 0],
            [torch.sin(rotation[2]), torch.cos(rotation[2]), 0],
            [0, 0, 1],
        ],
        dtype=graph.x.dtype,
        device=graph.x.device,
    )
    augmented_graph = graph.clone()
    assert augmented_graph.x is not None
    node_features_to_rotate = [
        ["x", "y", "z"],
        ["vel_x", "vel_y", "vel_z"],
        ["acceleration_x", "acceleration_y", "acceleration_z"],
    ]

    rotation_matrix = torch.matmul(rot_matrix_y, rot_matrix_x)
    rotation_matrix = torch.matmul(rot_matrix_z, rotation_matrix)

    for features in node_features_to_rotate:
        idxs = [config["node_features"][feature] for feature in features]
        augmented_graph.x[:, idxs] = torch.matmul(graph.x[:, idxs], rotation_matrix.T)

    augmented_graph.x1 = torch.matmul(graph.x1, rotation_matrix.T)
    augmented_graph.priors = torch.matmul(graph.priors, rotation_matrix.T)
    augmented_graph.priors_pos = torch.matmul(graph.priors_pos, rotation_matrix.T)
    augmented_graph.trajectories = torch.matmul(graph.trajectories, rotation_matrix.T)

    if graph.edge_index is not None and graph.edge_attr is not None:
        source, target = graph.edge_index
        augmented_graph.edge_attr[:, config["edge_features"]["delta_x"]] = (  # type: ignore
            augmented_graph.x[source, config["node_features"]["x"]]
            - augmented_graph.x[target, config["node_features"]["x"]]
        )
        augmented_graph.edge_attr[:, config["edge_features"]["delta_y"]] = (  # type: ignore
            augmented_graph.x[source, config["node_features"]["y"]]
            - augmented_graph.x[target, config["node_features"]["y"]]
        )
        augmented_graph.edge_attr[:, config["edge_features"]["delta_z"]] = (  # type: ignore
            augmented_graph.x[source, config["node_features"]["z"]]
            - augmented_graph.x[target, config["node_features"]["z"]]
        )
        augmented_graph.edge_attr[:, config["edge_features"]["delta_vel_x"]] = (  # type: ignore
            augmented_graph.x[source, config["node_features"]["vel_x"]]
            - augmented_graph.x[target, config["node_features"]["vel_x"]]
        )
        augmented_graph.edge_attr[:, config["edge_features"]["delta_vel_y"]] = (  # type: ignore
            augmented_graph.x[source, config["node_features"]["vel_y"]]
            - augmented_graph.x[target, config["node_features"]["vel_y"]]
        )
        augmented_graph.edge_attr[:, config["edge_features"]["delta_vel_z"]] = (  # type: ignore
            augmented_graph.x[source, config["node_features"]["vel_z"]]
            - augmented_graph.x[target, config["node_features"]["vel_z"]]
        )

    return augmented_graph


def make_graph(
    positions: Tensor,
    node_features: Tensor,
    config: dict,
    frame_batch: Tensor | None = None,
    batch: Tensor | None = None,
    context: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Construct features and edges from trajectory data using per timestep disconnected nodes
    Args:
        positions: Tensor of shape (batch_size * n_frames * n_trajectories, 2)
        node_features: Tensor of shape (batch_size * n_frames * n_trajectories, 2)
        config: Configuration dictionary
        frame_batch: Tensor of shape (batch_size * n_frames * n_trajectories,)
        batch: Tensor of shape (batch_size * n_frames * n_trajectories,)
    Returns:
        edge_index: Tensor of shape (2, n_edges)
        edge_attr: Tensor of shape (n_edges, n_features)"""

    if frame_batch is None:
        frames, batch_size = config["time_window"], config["batch_size"]
        trajs = positions.shape[0] // frames // batch_size
        frame_batches = torch.arange(frames, device=positions.device).repeat_interleave(trajs)
        frame_batch = frame_batches.unsqueeze(0).repeat(batch_size, 1).flatten()
    frame_batch = (
        frame_batch + (batch * config["time_window"]) if batch is not None else frame_batch
    )

    if config["knn"] is not None and config["radius_graph"] is None:
        e_index = knn_graph(positions.detach(), k=config["knn"], batch=frame_batch)
    elif config["knn"] is None and config["radius_graph"] is not None:
        e_index = radius_graph(positions.detach(), r=config["radius_graph"], batch=frame_batch)
    elif config["knn"] is not None and config["radius_graph"] is not None:
        e_index = radius_graph(
            positions.detach(),
            r=config["radius_graph"],
            batch=frame_batch,
            max_num_neighbors=config["knn"] - 1,
        )

    edge_attr = calculate_edge_features(
        positions, node_features, e_index, config, context=context
    )

    return e_index, edge_attr


def calculate_node_features(
    positions: Tensor,
    velocities: Tensor,
    accelerations: Tensor,
    config: dict,
    time_axis_idx: Tensor | None = None,
    context: Tensor | None = None,
) -> Tensor:
    """Calculate node features from velocities
    Args:
        positions: Tensor of shape (batch_size * num_frames * num_nodes, 2)
        velocities: Tensor of shape (batch_size * num_frames * num_nodes, 2)
        accelerations: Tensor of shape (batch_size * num_frames * num_nodes, 2)
    Returns:
        node_feats: Tensor of shape (batch_size, num_frames,num_nodes, n_feat)"""

    feat, n_feat = config["node_features"], len(config["node_features"].keys())

    # if velocities.dim() == 3:  # Add batch dimension if not present
    #     positions = positions.unsqueeze(0)
    #     velocities = velocities.unsqueeze(0)
    #     accelerations = accelerations.unsqueeze(0)

    node_feats = torch.zeros(
        positions.size(0),
        n_feat,
        device=velocities.device,
        dtype=torch.float16 if config["use_fp16"] else torch.float32,
    )

    # ang_feats = ["angular_velocity", "angular_velocity_sine", "angular_velocity_cosine"]
    # if any([f in feat for f in ang_feats]):
    #     θ = torch.arctan2(
    #         velocities[:, :, :, 1], velocities[:, :, :, 0]
    #     )  # Direction of velocity
    #     ω = torch.diff(θ, dim=1)  # Angular velocity
    #     ω = torch.cat([ω, ω[:, -1].unsqueeze(1)], dim=1)

    if "x" in feat:
        x = positions[..., 0]
        if config["predict_velocity"] or config["predict_acceleration"]:
            # Scale x with min and max range
            x = (x - config["x_range"][0]) / (config["x_range"][1] - config["x_range"][0])
        node_feats[:, feat["x"]] = x
    if "y" in feat:
        y = positions[..., 1]
        if config["predict_velocity"] or config["predict_acceleration"]:
            # Scale y with min and max range
            y = (y - config["y_range"][0]) / (config["y_range"][1] - config["y_range"][0])
        node_feats[:, feat["y"]] = y
    if "z" in feat:
        z = positions[..., 2]
        if config["predict_velocity"] or config["predict_acceleration"]:
            # Scale y with min and max range
            z = (z - config["z_range"][0]) / (config["z_range"][1] - config["z_range"][0])
        node_feats[:, feat["z"]] = z
    if "vel_x" in feat:
        node_feats[:, feat["vel_x"]] = velocities[..., 0]
    if "vel_y" in feat:
        node_feats[:, feat["vel_y"]] = velocities[..., 1]
    if "vel_z" in feat:
        node_feats[:, feat["vel_z"]] = velocities[..., 2]
    if "velocity_magnitude" in feat:
        velocity_magnitude = torch.norm(velocities, dim=-1)
        node_feats[:, feat["velocity_magnitude"]] = velocity_magnitude
    if "acceleration_x" in feat:
        node_feats[:, feat["acceleration_x"]] = accelerations[..., 0]
    if "acceleration_y" in feat:
        node_feats[:, feat["acceleration_y"]] = accelerations[..., 1]
    if "acceleration_z" in feat:
        node_feats[:, feat["acceleration_z"]] = accelerations[..., 2]
    if "acceleration_magnitude" in feat:
        acceleration_magnitude = torch.norm(accelerations, dim=-1)
        node_feats[:, feat["acceleration_magnitude"]] = acceleration_magnitude

    # if "sliding_velocity" in feat:
    #     trailing_window = velocities.shape[1] // 4
    #     vels = torch.norm(velocities, dim=-1).permute(0, 2, 1)
    #     sliding_velocity = F.avg_pool1d(
    #         vels, kernel_size=trailing_window, stride=1, padding=trailing_window // 2
    #     )
    #     sliding_velocity = sliding_velocity.permute(0, 2, 1)
    #     node_feats[:, feat["sliding_velocity"]] = sliding_velocity
    # if "sliding_acceleration" in feat:
    #     trailing_window = velocities.shape[1] // 4
    #     acc = acceleration_magnitude.permute(0, 2, 1)
    #     sliding_acceleration = F.avg_pool1d(
    #         acc, kernel_size=trailing_window, stride=1, padding=trailing_window // 2
    #     )
    #     sliding_acceleration = sliding_acceleration.permute(0, 2, 1)
    #     node_feats[:, feat["sliding_acceleration"]] = sliding_acceleration
    # if "angular_velocity_sine" in feat:
    #     sine = torch.sin(ω)
    #     node_feats[:, :, :, feat["angular_velocity_sine"]] = sine
    # if "angular_velocity_cosine" in feat:
    #     cosine = torch.cos(ω)
    # node_feats[:, :, :, feat["angular_velocity_cosine"]] = cosine
    # if "angular_velocity" in feat:
    #     node_feats[:, :, :, feat["angular_velocity"]] = ω
    # if "normalized_angular_velocity" in feat:
    #     node_feats[:, :, :, feat["normalized_angular_velocity"]] = ω / (
    #         velocity_magnitude + 1e-8
    #     )
    if "local_density" in feat:
        pos = (
            positions[time_axis_idx]
            if time_axis_idx is not None
            else positions.view(config["time_window"], -1, 2).transpose(0, 1)
        )
        local_density = calculate_local_density(pos, config)
        if time_axis_idx is not None:
            density = torch.zeros_like(positions[:, :1])  # type: ignore
            idx = time_axis_idx.flatten().unsqueeze(-1).expand(-1, density.size(-1))
            density = density.scatter(0, idx, local_density.flatten().view(-1, 1)).squeeze(-1)
        else:
            density = local_density.transpose(0, 1).flatten(0, 1)
        node_feats[:, feat["local_density"]] = density

    if context is not None and "class" in feat:
        node_feats[:, feat["class"]] = context[:, 0]

    if context is not None and "scene" in feat and "class" in feat:
        node_feats[:, feat["scene"]] = context[:, 1]

    if context is not None and "scene" in feat and "class" not in feat:
        node_feats[:, feat["scene"]] = context[:, 0]

    if context is not None and "charge" in feat:
        node_feats[:, feat["charge"]] = context[:, 0]

    if context is not None and "force_x" in feat:
        node_feats[:, feat["force_x"]] = context[:, 0]

    if context is not None and "force_y" in feat:
        node_feats[:, feat["force_y"]] = context[:, 1]

    if context is not None and "force_z" in feat:
        node_feats[:, feat["force_z"]] = context[:, 2]

    if context is not None and "atom_0" in feat:
        # charge_power = 2
        # charge_tensor = (atom_type.unsqueeze(-1) / 10).pow(
        #     torch.arange(charge_power + 1.0, device=atom_type.device)
        # )
        # charge_tensor = charge_tensor.view(atom_type.shape + (1, charge_power + 1))
        # atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
        for i in range(10):
            node_feats[:, feat[f"atom_{i}"]] = context[:, i]

    return node_feats


def calculate_local_density(positions: Tensor, config: dict) -> Tensor:
    """
    Compute local cell density using Gaussian kernel density estimation.

    Args:
        positions: Tensor of shape (batch * num_cells,frames, 2) containing node positions
        config: dict
    Returns:
        densities: Tensor of shape (batch * num_cells,frames) containing local density estimates
    """
    pos = positions.transpose(0, 1)
    # Compute pairwise distances between all cells
    pos_i = pos.unsqueeze(2)
    pos_j = pos.unsqueeze(1)

    # Compute squared distances
    squared_distances = torch.norm(pos_i - pos_j, dim=-1)  # (frames, nodes, nodes)

    # Apply Gaussian kernel
    kernel_values = torch.exp(-squared_distances / (2 * config["dist_kernel"] ** 2))

    # Sum over all neighboring cells (excluding self-interaction)
    self_mask = torch.eye(pos.shape[1], device=positions.device)[None, :, :]
    kernel_values = kernel_values * (1 - self_mask)

    # Compute density by summing kernel values
    densities = torch.sum(kernel_values, dim=-1, keepdim=True)

    # Normalize densities
    # densities = densities / torch.max(densities, dim=1, keepdim=True)[0]

    return densities.transpose(0, 1).squeeze(-1)


def pad_graph_batches(
    x1: Tensor, x1_pos: Tensor, batch: Tensor, config: dict, size: int
) -> tuple[Tensor, Tensor]:
    """Pad graph batches with zeros to match amount of nodes in each batch
    Args:
        x1: Tensor of shape (batch * frames * nodes, 2)
        x1_pos: Tensor of shape (batch * frames * nodes, 2)
        batch: Tensor of shape (batch * frames * nodes,)
        Returns: Padded x1 and x1_pos of shape (batch, frames, max_trajs, 2)"""

    batch_size, n_frames = size, config["time_window"]
    node_counts = torch.bincount(batch, minlength=batch_size)  # type: ignore
    traj_counts = node_counts // n_frames
    max_trajs = max(traj_counts)
    pad_x1 = torch.zeros(batch_size, n_frames, max_trajs, config["dims"], device=x1.device)  # type: ignore
    pad_x1_pos = torch.zeros(batch_size, n_frames, max_trajs, config["dims"], device=x1.device)  # type: ignore
    # Seperate padded nodes spatially
    pad_x1_pos += 1000

    idx = 0
    for b in range(batch_size):  # type: ignore
        x1_batch = x1[idx : idx + node_counts[b]]
        x1_pos_batch = x1_pos[idx : idx + node_counts[b]]
        pad_x1[b, :, : traj_counts[b]] = x1_batch.view(n_frames, -1, config["dims"])
        pad_x1_pos[b, :, : traj_counts[b]] = x1_pos_batch.view(n_frames, -1, config["dims"])
        idx += node_counts[b]

    return pad_x1, pad_x1_pos


def get_dynamic_graph_batches(
    xt: Tensor,
    ut: Tensor,
    data: Data,
    x0_pos: Tensor,
    x1_pos: Tensor,
    config: dict,
    masks: Tensor | None,
    x0_idx=None,
    x1_idx=None,
) -> tuple[Data, Tensor, Tensor]:
    """Construct dynamic graph batches
    Args:
    xt: Tensor of shape (batch, num_frames * max_num_nodes, dims)
    ut: Tensor of shape (batch, num_frames * max_num_nodes, dims)
    data: Original data object
    x0_pos: Tensor of shape (batch, num_frames * max_num_nodes, dims)
    x1_pos: Tensor of shape (batch, num_frames * max_num_nodes,dims)
    config: Configuration dictionary
    masks: Tensor of shape (batch, num_frames * max_num_nodes, #features)
    x0_idx: Index of x0 nodes
    x1_idx: Index of x1 nodes
    Returns:
    xt_graph: BatchData object containing dynamic graphs
    ut_flat: Tensor of shape (batch, num_frames * max_num_nodes, dims)
    xt_flat
    """
    n_frames, batch_size = config["time_window"], xt.shape[0]
    pad_size = xt.shape[1] // config["time_window"]

    # Reorder data based on OT coupling
    if x0_idx is not None and x1_idx is not None:
        xt_graph = Batch.from_data_list(data.index_select(x1_idx))
    else:
        xt_graph = data

    xt = xt.view(batch_size, n_frames, pad_size, config["dims"])
    ut = ut.view(batch_size, n_frames, pad_size, config["dims"])
    x0_pos = x0_pos.view(batch_size, n_frames, pad_size, config["dims"])
    x1_pos = x1_pos.view(batch_size, n_frames, pad_size, config["dims"])

    if config["predict_acceleration"]:
        all_pos = x1_pos
        start_pos = all_pos[:, 0].unsqueeze(1)
        start_vel = all_pos[:, 1].unsqueeze(1) - start_pos
        accs = xt
        vels = torch.cat([start_vel, accs], dim=1).cumsum(dim=1)[:, :-1]
        pos = torch.cat([start_pos, vels], dim=1).cumsum(dim=1)[:, :-1]
    elif config["predict_velocity"]:
        all_pos = x1_pos
        start_pos = all_pos[:, 0].unsqueeze(1)
        vels = xt
        pos = torch.cat([start_pos, vels], dim=1).cumsum(dim=1)[:, :-1]
        accs = torch.diff(vels, dim=1)
        accs = torch.cat([accs, accs[:, -1:]], dim=1)
    else:
        pos = xt
        vels = torch.diff(pos, dim=1)
        vels = torch.cat([vels, vels[:, -1:]], dim=1)
        accs = torch.diff(vels, dim=1)
        accs = torch.cat([accs, accs[:, -1:]], dim=1)

    # Use trajectory counts to remove padding
    node_counts = torch.unique(xt_graph.batch, return_counts=True)[1]  # type: ignore
    pad_mask = torch.arange(pad_size, device=xt.device).expand(xt.shape[0], n_frames, pad_size)
    pad_mask = pad_mask < (node_counts.view(-1, 1, 1) // n_frames)
    xt_flat = xt[pad_mask].view(-1, config["dims"])
    ut_flat = ut[pad_mask].view(-1, config["dims"])
    pos = pos[pad_mask].view(-1, config["dims"])
    vels = vels[pad_mask].view(-1, config["dims"])
    accs = accs[pad_mask].view(-1, config["dims"])
    if masks is not None:
        masks = masks.view(batch_size, n_frames, pad_size, -1)
        masks_flat = masks[pad_mask].view(-1, masks.shape[-1])
    prior_pos_flat = x0_pos[pad_mask].view(-1, config["dims"])

    time_axis_idx = get_time_axis_idx(xt_graph.batch, config)  # type: ignore
    xt_graph.x = calculate_node_features(  # type: ignore
        pos,
        vels,
        accs,
        config,
        time_axis_idx=time_axis_idx,
        context=xt_graph.context,  # type: ignore
    )

    pos, vels, accs = (
        pos.view(-1, config["dims"]),
        vels.view(-1, config["dims"]),
        accs.view(-1, config["dims"]),
    )

    if config["use_dynamic_edges"]:
        knn_batches = torch.arange(n_frames, device=xt.device).repeat_interleave(pad_size)
        knn_batches = knn_batches.unsqueeze(0).repeat(batch_size, 1)
        knn_batches = knn_batches.view(batch_size, n_frames, pad_size, -1)[pad_mask].flatten()
        e_indexes, e_attrs = make_graph(
            pos,
            xt_graph.x,  # type: ignore
            config,
            frame_batch=knn_batches,
            batch=data.batch,  # type: ignore
            context=xt_graph.context,  # type: ignore
        )
        xt_graph.edge_index = e_indexes  # type: ignore
        xt_graph.edge_attr = e_attrs  # type: ignore
        xt_graph.dist = e_attrs[:, :1]  # type: ignore
    else:
        xt_graph.edge_index = torch.tensor([], device=xt.device)  # type: ignore
        xt_graph.edge_attr = torch.tensor([], device=xt.device)  # type: ignore

    xt_graph.priors_pos = prior_pos_flat  # type: ignore
    xt_graph.mask = masks_flat  # type: ignore

    return xt_graph, xt_flat, ut_flat  # type: ignore


def velocity_to_position(
    x1_positions: Tensor, velocities: Tensor, time_idx: Tensor, config: dict
) -> Tensor:
    """Integrates velocities to obtain positions, given start position and conditioning.
    Args:
        start_position: Tensor of shape (batch * frames * trajs, dims)
        velocities: Tensor of shape (batch * frames * trajs, frames, dims)
        time_idx: Tensor of shape (batch * trajs, frames)
    Returns:
        positions: Tensor of shape (batch * frames * trajs, dims)
    """
    f = config["cond_frames"]
    start_position, vels = (
        x1_positions[time_idx][:, f - 1 : f],
        velocities[time_idx][:, f - 1 : -1],
    )
    positions = torch.cat([start_position, vels], dim=1).cumsum(dim=1)
    positions = torch.cat([x1_positions[time_idx][:, : f - 1], positions], dim=1)
    flat_positions = torch.zeros_like(velocities)
    positions = positions.flatten(0, 1)
    idx = time_idx.flatten().unsqueeze(-1).expand(-1, flat_positions.size(-1))
    flat_positions = flat_positions.scatter(0, idx, positions)
    return flat_positions


def position_to_velocity(
    positions: Tensor, x1_positions: Tensor, time_idx: Tensor, config: dict
) -> Tensor:
    """Calculates velocities from positions by taking differences, with conditioning
    Args:
        positions: Tensor of shape (batch * frames * trajs, dims)
        x1_positions: Tensor of shape (batch * frames * trajs, dims)
        time_idx: Tensor of shape (batch * trajs, frames)
    Returns:
        velocities: Tensor of shape (batch * frames * trajs, frames, dims)
    """
    f = config["cond_frames"]
    pos = torch.cat([x1_positions[time_idx][:, :f], positions[time_idx][:, f:]], dim=1)
    vels = torch.diff(pos, dim=1)
    vels = torch.cat([vels, vels[:, -1:]], dim=1).flatten(0, 1)
    flat_vels = torch.zeros_like(positions)
    idx = time_idx.flatten().unsqueeze(-1).expand(-1, flat_vels.size(-1))
    flat_vels = flat_vels.scatter(0, idx, vels)
    return flat_vels


def velocity_to_acceleration(
    velocities: Tensor, x1_velocities: Tensor, time_idx: Tensor, config: dict
) -> Tensor:
    """Calculates accelerations from velocities by taking differences, with conditioning
    Args:
        velocities: Tensor of shape (batch * frames * trajs, frames, dims)
        x1_velocities: Tensor of shape (batch * frames * trajs, frames, dims)
        time_idx: Tensor of shape (batch * trajs, frames)
    Returns:
        accelerations: Tensor of shape (batch * frames * trajs, frames, dims)
    """
    f = config["cond_frames"]
    vels = torch.cat(
        [x1_velocities[time_idx][:, : f - 1], velocities[time_idx][:, f - 1 :]], dim=1
    )
    accs = torch.diff(vels, dim=1)
    accs = torch.cat([accs, accs[:, -1:]], dim=1).flatten(0, 1)
    flat_accs = torch.zeros_like(velocities)
    idx = time_idx.flatten().unsqueeze(-1).expand(-1, flat_accs.size(-1))
    flat_accs = flat_accs.scatter(0, idx, accs)
    return flat_accs


def get_time_axis_idx(batch: Tensor, config: dict) -> Tensor:
    nodes_per_batch = scatter_add(torch.ones_like(batch), batch, dim=0)  # type: ignore
    trajs_per_batch = nodes_per_batch // config["time_window"]
    conv_batch_dim = batch.shape[0] // config["time_window"]  # type: ignore

    traj_p_b_idx = torch.cat(
        [torch.zeros(1, device=batch.device), trajs_per_batch.cumsum(dim=0)[:-1]]  # type: ignore
    ).repeat_interleave(trajs_per_batch)
    indices = torch.arange(conv_batch_dim, device=batch.device) - traj_p_b_idx  # type: ignore
    indices = indices.view(-1, 1).repeat(1, config["time_window"])

    offsets = (
        trajs_per_batch.repeat_interleave(trajs_per_batch)
        .view(-1, 1)
        .repeat(1, config["time_window"])
    )
    offsets = offsets * torch.arange(config["time_window"], device=batch.device)  # type: ignore

    batch_sizes = torch.cat(
        [torch.zeros(1, device=batch.device), nodes_per_batch.cumsum(dim=0)[:-1]]  # type: ignore
    )  # type: ignore

    batch_offsets = batch_sizes.repeat_interleave(
        nodes_per_batch // config["time_window"]
    ).view(-1, 1)  # type: ignore

    indices = (indices + offsets + batch_offsets).to(torch.long)

    return indices


def merge_time_axis(x: Tensor, time_axis_idx: Tensor) -> Tensor:
    """Transforms shape of (batch * nodes, frames, x) to (batch * nodes * frames, x)
    given time_axis_idx of shape (batch * nodes, frames)"""
    flat_x = torch.zeros_like(x.flatten(0, 1))
    idx = time_axis_idx.flatten().unsqueeze(-1).expand(-1, x.size(-1))
    flat_x = flat_x.scatter(0, idx, x.flatten(0, 1))
    return flat_x


class PedestriansDataset(Dataset):
    def __init__(self, config):
        self.config = config

        if config["do_preprocess"]:
            pedestrians = preprocess_pedestrians(config)
            samples = [torch.from_numpy(sample) for sample in pedestrians]
        else:
            if os.path.exists(config["processed_path"]) and not config["process_graphs"]:
                all_graphs = torch.load(config["processed_path"], weights_only=False)
                data_size = min(len(all_graphs), config["max_samples"])
                data_samples = random.sample(range(len(all_graphs)), data_size)
                # data_samples = range(len(all_graphs))
                self.graphs = [all_graphs[i] for i in data_samples]
                num_trajs = sum(
                    [len(graph.x) // config["time_window"] for graph in self.graphs]
                )
                print(f"\nLoaded {len(self.graphs)} graphs with {num_trajs} trajectories.\n")
                del all_graphs
                return
            else:
                pedestrians = np.load(config["save_path"], allow_pickle=True)
                samples = [torch.from_numpy(sample) for sample in pedestrians]

        samples = [
            trajectory.to(
                dtype=torch.float16 if config["use_fp16"] else torch.float32,
                # device="cuda" if torch.cuda.is_available() else "cpu",
            ).transpose(0, 1)
            for trajectory in samples
        ]  # (frames, trajs, 2)
        velocities = [torch.diff(traj, dim=0) for traj in samples]
        velocities = [torch.cat([vel, vel[-1:]], dim=0) for vel in velocities]
        accelerations = [torch.diff(vel, dim=0) for vel in velocities]
        accelerations = [torch.cat([acc, acc[-1:]], dim=0) for acc in accelerations]

        # Filter trajectories
        filtered_samples, filtered_velocities, filtered_accelerations = [], [], []
        for i, sample in enumerate(samples):
            mean_vel_magnitude = torch.mean(torch.norm(velocities[i], dim=-1), dim=0)
            mask = mean_vel_magnitude > config["min_velocity_filter"]
            if sum(mask) >= config["min_num_trajs_filter"]:
                filtered_samples.append(sample[:, mask])
                filtered_velocities.append(velocities[i][:, mask])
                filtered_accelerations.append(accelerations[i][:, mask])

        self.graphs = []
        print(
            f"\n #Trajectories in dataset: {sum([sample.shape[1] for sample in filtered_samples])}\n"
        )
        data_size = (
            len(filtered_samples)
            if config["max_samples"] is None
            else min(len(filtered_samples), config["max_samples"])
        )
        data_samples = random.sample(range(len(filtered_samples)), data_size)
        # data_samples = range(data_size)
        for i in tqdm(data_samples, desc="Constructing graphs..."):
            trajs = filtered_samples[i][:, :, 0:2]
            vels = filtered_velocities[i][:, :, 0:2]
            accs = filtered_accelerations[i][:, :, 0:2]
            if filtered_samples[i].shape[-1] > 2:
                context = filtered_samples[i][:, :, 2:]
            else:
                context = torch.tensor([], device=trajs.device)

            features = calculate_node_features(
                trajs.flatten(0, 1),
                vels.flatten(0, 1),
                accs.flatten(0, 1),
                config,
                context=context.flatten(0, 1),
            )

            batch_length = trajs.flatten(0, 1).shape[0]
            batch = torch.zeros(batch_length, dtype=torch.long, device=trajs.device)
            priors, priors_pos, mask = sample_prior(trajs, batch, config, inference=False)

            knn_batch = torch.arange(config["time_window"], device=trajs.device)
            knn_batch = knn_batch.repeat_interleave(trajs.shape[1])
            edge_index, edge_attr = make_graph(
                trajs.flatten(0, 1), features, config, frame_batch=knn_batch
            )

            priors = priors.flatten(0, 1)
            priors_pos = priors_pos.flatten(0, 1)
            trajs = trajs.flatten(0, 1)
            vels = vels.flatten(0, 1)
            accs = accs.flatten(0, 1)
            if config["predict_acceleration"]:
                x1 = accs
            elif config["predict_velocity"]:
                x1 = vels
            else:
                x1 = trajs

            mask = mask.flatten(0, 1) if mask is not None else None

            dynamic_graph = Data(
                x=features,  # Node features (#frames * #nodes, node_dim)
                x1=x1,
                edge_index=edge_index,  # Edge indexes (2, #edges)
                edge_attr=edge_attr,  # Edge features (#edges, edge_dim)
                priors=priors,  # FM priors  (#frames * #nodes, node_dim)
                priors_pos=priors_pos,  # FM priors positions (#frames * #nodes, node_dim)
                trajectories=trajs,  # Node trajectories (#frames * #nodes, node_dim)
                # velocities=vels,  # Node velocities (#frames * #nodes, node_dim)
                mask=mask,  # Masks for conditioning (#frames * #nodes)
                # y=global_features,  # Global features (1, n_feat)
                context=context.flatten(0, 1),
            )

            self.graphs.append(dynamic_graph)

        print("Saving graphs...")
        torch.save(self.graphs, config["processed_path"])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> Data:
        return self.graphs[idx]


class NBodyDataset(Dataset):
    def __init__(self, config):
        self.config = config

        if config["type"] == "charged":
            loc, vel, charges, edges = [], [], [], []
            for info in ["loc", "vel", "charges", "edges"]:
                for split in ["train", "valid", "test"]:
                    suffix = "charged5_initvel1"
                    data = np.load(config["data_paths"] + f"/{info}_{split}_{suffix}.npy")
                    if info == "loc":
                        loc.append(data)
                    elif info == "vel":
                        vel.append(data)
                    elif info == "charges":
                        charges.append(data)
                    elif info == "edges":
                        edges.append(data)

            charges = torch.from_numpy(np.concatenate(charges, axis=0)).float()
            edges = torch.from_numpy(np.concatenate(edges, axis=0)).long()
            # (#samples, #frames, trajectories, 3)
            loc = torch.from_numpy(np.concatenate(loc, axis=0)).float().transpose(2, 3)
        elif config["type"] == "gravity":
            loc = []
            for info in ["loc"]:
                for split in ["train", "valid", "test"]:
                    suffix = "gravity10_initvel1"
                    data = np.load(config["data_paths"] + f"/{info}_{split}_{suffix}.npy")
                    if info == "loc":
                        loc.append(data)
            # (#samples, #frames, trajectories, 3)
            loc = torch.from_numpy(np.concatenate(loc, axis=0)).float()
        elif config["type"] == "springs":
            loc, edges = [], []
            for info in ["loc", "edges"]:
                for split in ["train", "valid", "test"]:
                    suffix = "springs5_initvel1"
                    data = np.load(config["data_paths"] + f"/{info}_{split}_{suffix}.npy")
                    if info == "loc":
                        loc.append(data)
                    elif info == "edges":
                        edges.append(data)
            # (#samples, #frames, trajectories, 3)
            loc = torch.from_numpy(np.concatenate(loc, axis=0)).float().transpose(2, 3)
            edges = torch.from_numpy(np.concatenate(edges, axis=0)).long()

        # Limit training size to match experimental setup
        data_size = 3000 + 2000 + 2000

        self.graphs = []
        print(f"\n #Trajectories in dataset: {loc.shape[0]}\n")
        data_samples = range(data_size)
        for i in tqdm(data_samples, desc="Constructing graphs..."):
            trajs = loc[i][: config["time_window"], :, 0:3]
            vels = torch.diff(trajs, dim=0)
            vels = torch.cat([vels, vels[-1:]], dim=0)
            accs = torch.diff(vels, dim=0)
            accs = torch.cat([accs, accs[-1:]], dim=0)
            if config["type"] == "charged":
                charge = (
                    charges[i].view(1, vels.shape[1], 1).repeat(config["time_window"], 1, 1)
                )
                context = charge.flatten(0, 1)
            elif config["type"] == "gravity":
                context = torch.Tensor([], device=vels.device)
            elif config["type"] == "springs":
                context = None

            features = calculate_node_features(
                trajs.flatten(0, 1),
                vels.flatten(0, 1),
                accs.flatten(0, 1),
                config,
                context=context,
            )

            batch_length = trajs.flatten(0, 1).shape[0]
            batch = torch.zeros(batch_length, dtype=torch.long, device=trajs.device)
            priors, priors_pos, mask = sample_prior(trajs, batch, config, inference=False)

            knn_batch = torch.arange(config["time_window"], device=trajs.device)
            knn_batch = knn_batch.repeat_interleave(trajs.shape[1])
            edge_index, edge_attr = make_graph(
                trajs.flatten(0, 1), features, config, frame_batch=knn_batch, context=context
            )

            if config["type"] == "springs":
                e = edge_index % 5
                springs = edges[i][e[0], e[1]]
                context = springs
                edge_attr[:, config["edge_features"]["spring"]] = springs

            priors = priors.flatten(0, 1)
            priors_pos = priors_pos.flatten(0, 1)
            trajs = trajs.flatten(0, 1)
            vels = vels.flatten(0, 1)
            accs = accs.flatten(0, 1)
            if config["predict_acceleration"]:
                x1 = accs
            elif config["predict_velocity"]:
                x1 = vels
            else:
                x1 = trajs

            mask = mask.flatten(0, 1) if mask is not None else None

            dynamic_graph = Data(
                x=features,  # Node features (#frames * #nodes, node_dim)
                x1=x1,
                edge_index=edge_index,  # Edge indexes (2, #edges)
                edge_attr=edge_attr,  # Edge features (#edges, edge_dim)
                priors=priors,  # FM priors  (#frames * #nodes, node_dim)
                priors_pos=priors_pos,  # FM priors positions (#frames * #nodes, node_dim)
                trajectories=trajs,  # Node trajectories (#frames * #nodes, node_dim)
                # velocities=vels,  # Node velocities (#frames * #nodes, node_dim)
                mask=mask,  # Masks for conditioning (#frames * #nodes)
                # y=global_features,  # Global features (1, n_feat)
                context=context,
            )

            self.graphs.append(dynamic_graph)

        print("Saving graphs...")
        torch.save(self.graphs, config["processed_path"])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> Data:
        return self.graphs[idx]


class MD17Dataset(Dataset):
    def __init__(self, config):
        self.config = config

        path = config["data_paths"] + f"/md17_{config['type']}.npz"

        data = np.load(path, allow_pickle=True)

        loc = torch.from_numpy(data["R"]).to(dtype=torch.float32)
        atoms = torch.from_numpy(data["z"])
        one_hot = F.one_hot(atoms.long(), 10).unsqueeze(0).to(dtype=torch.float32)
        context = one_hot.repeat(config["time_window"], 1, 1).flatten(0, 1)

        loc = loc[::10]  # Subsample data

        # loc = torch.split(loc, config["time_window"])[:-1]
        # Use a sliding window of 30 time steps with x steps in between
        split_loc = []
        step = 10
        for i in range(0, loc.shape[0] - config["time_window"] + 1, step):
            split_loc.append(loc[i : i + config["time_window"], :, 0:3])

        self.graphs = []
        data_size = min(len(split_loc), config["max_samples"])
        data_samples = range(data_size)  # type: ignore
        for i in tqdm(data_samples, desc="Constructing graphs..."):
            trajs = split_loc[i][: config["time_window"], :, 0:3]
            vels = torch.diff(trajs, dim=0)
            vels = torch.cat([vels, vels[-1:]], dim=0)
            accs = torch.diff(vels, dim=0)
            accs = torch.cat([accs, accs[-1:]], dim=0)

            features = calculate_node_features(
                trajs.flatten(0, 1),
                vels.flatten(0, 1),
                accs.flatten(0, 1),
                config,
                context=context,
            )

            batch_length = trajs.flatten(0, 1).shape[0]
            batch = torch.zeros(batch_length, dtype=torch.long, device=trajs.device)
            priors, priors_pos, mask = sample_prior(trajs, batch, config, inference=False)

            knn_batch = torch.arange(config["time_window"], device=trajs.device)
            knn_batch = knn_batch.repeat_interleave(trajs.shape[1])
            edge_index, edge_attr = make_graph(
                trajs.flatten(0, 1), features, config, frame_batch=knn_batch
            )

            priors = priors.flatten(0, 1)
            priors_pos = priors_pos.flatten(0, 1)
            trajs = trajs.flatten(0, 1)
            vels = vels.flatten(0, 1)
            accs = accs.flatten(0, 1)
            if config["predict_acceleration"]:
                x1 = accs
            elif config["predict_velocity"]:
                x1 = vels
            else:
                x1 = trajs

            mask = mask.flatten(0, 1) if mask is not None else None

            dynamic_graph = Data(
                x=features,  # Node features (#frames * #nodes, node_dim)
                x1=x1,
                edge_index=edge_index,  # Edge indexes (2, #edges)
                edge_attr=edge_attr,  # Edge features (#edges, edge_dim)
                priors=priors,  # FM priors  (#frames * #nodes, node_dim)
                priors_pos=priors_pos,  # FM priors positions (#frames * #nodes, node_dim)
                trajectories=trajs,  # Node trajectories (#frames * #nodes, node_dim)
                # velocities=vels,  # Node velocities (#frames * #nodes, node_dim)
                mask=mask,  # Masks for conditioning (#frames * #nodes)
                # y=global_features,  # Global features (1, n_feat)
                context=context,
            )

            self.graphs.append(dynamic_graph)

        print("Saving graphs...")
        torch.save(self.graphs, config["processed_path"])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx) -> Data:
        return self.graphs[idx]


class GraphsDataModule(L.LightningDataModule):
    def __init__(self, config, is_training=False):
        super().__init__()
        self.config = config
        self.is_training = is_training
        if "pedestrians" in config["data_paths"] or isinstance(config["data_paths"], list):
            self.dataset = PedestriansDataset(config)
            if "ETH" in config["data_paths"]:
                # For ETH UCY, we seperate one dataset and train on the rest
                self.train_val_data, self.test_data = [], []
                for i in range(len(self.dataset)):
                    # Do train-test splitting based on scene, as leave-one-out
                    context = self.dataset[i].context
                    if int(context[0]) != self.config["type"]:
                        self.train_val_data.append(self.dataset[i])
                    else:
                        self.test_data.append(self.dataset[i])
        elif "nbody" in config["data_paths"]:
            self.dataset = NBodyDataset(self.config)
            train_val_size = 3000 + 2000
            test_size = 2000
            self.train_val_data = self.dataset[:train_val_size]
            self.test_data = self.dataset[train_val_size : train_val_size + test_size]
        elif "md17" in config["data_paths"]:
            self.dataset = MD17Dataset(self.config)
            test_split = int(config["test_split"] * len(self.dataset))
            self.train_val_data = self.dataset[:-test_split]
            self.test_data = self.dataset[-test_split:]
        else:
            ValueError("Dataset not found")

    def setup(self, stage):
        if stage == "fit":
            if "nbody" in self.config["data_paths"]:
                train_data = self.train_val_data[:3000]  # type: ignore
                val_data = self.train_val_data[3000:5000]  # type: ignore
            elif "md17" in self.config["data_paths"]:
                val_split = int(self.config["val_split"] * len(self.train_val_data))
                train_data = self.train_val_data[:-val_split]  # type: ignore
                val_data = self.train_val_data[-val_split:]  # type: ignore
            else:
                train_data, val_data = random_split(
                    self.train_val_data,  # type: ignore
                    lengths=[1 - self.config["val_split"], self.config["val_split"]],
                )

            # Do augmentations, after splitting
            augmented_graphs_train, augmented_graphs_val = [], []
            for augmentation in tqdm(self.config["augmentations"], desc="Augmenting data"):
                for graph in train_data:  # type: ignore
                    if augmentation == "rotate":
                        if self.config["dims"] == 2:
                            random_angle = torch.rand(1) * 2 * np.pi
                            augmented_graph = augment_by_rotation_2d(
                                graph.clone(),  # type: ignore
                                random_angle,
                                self.config,
                            )
                        elif self.config["dims"] == 3:
                            random_angles = torch.rand(3) * 2 * np.pi
                            augmented_graph = augment_by_rotation_3d(
                                graph.clone(),  # type: ignore
                                random_angles,
                                self.config,
                            )
                        augmented_graphs_train.append(augmented_graph)
                for graph in val_data:  # type: ignore
                    if augmentation == "rotate":
                        if self.config["dims"] == 2:
                            random_angle = torch.rand(1) * 2 * np.pi
                            augmented_graph = augment_by_rotation_2d(
                                graph.clone(),  # type: ignore
                                random_angle,
                                self.config,
                            )
                        elif self.config["dims"] == 3:
                            random_angles = torch.rand(3) * 2 * np.pi
                            augmented_graph = augment_by_rotation_3d(
                                graph.clone(),  # type: ignore
                                random_angles,
                                self.config,
                            )
                        augmented_graphs_val.append(augmented_graph)
            if len(augmented_graphs_train) > 0:
                print(
                    f"Augmented train/val data with {len(augmented_graphs_train) + len(augmented_graphs_val)} extra graphs by rotating.\n"
                )

            self.train_data = list(train_data) + augmented_graphs_train  # type: ignore
            self.val_data = list(val_data) + augmented_graphs_val  # type: ignore

            num_nodes = sum([graph.x.shape[0] for graph in self.train_data])  # type: ignore
            num_edges = sum([graph.edge_index.shape[1] for graph in self.train_data])  # type: ignore
            print(
                f"\n Total number of nodes: {num_nodes}, number of edges: {num_edges} in training data\n"
            )

            nodes_per_graph = num_nodes // len(self.train_data)
            edges_per_graph = num_edges // len(self.train_data)
            print(
                f"Average number of nodes per graph: {nodes_per_graph}, average number of edges per graph: {edges_per_graph}\n"
            )

    def train_dataloader(self):
        return loader.DataLoader(
            self.train_data,  # type: ignore
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        return loader.DataLoader(
            self.val_data,  # type: ignore
            batch_size=self.config["batch_size"],  # self.config["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=False,
        )

    def test_dataloader(self):
        return loader.DataLoader(
            self.test_data,  # type: ignore
            batch_size=len(self.test_data),
            num_workers=0,
            shuffle=False,
            drop_last=True,
        )
