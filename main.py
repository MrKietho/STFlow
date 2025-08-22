import torch
from torch import Tensor
from evaluation import compute_ade
from utils.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import WandbLogger
from data import GraphsDataModule
from lightning.pytorch.utilities import disable_possible_user_warnings
from models import DynamicGraphNetwork
from torch_geometric.data import Data, Batch
import time
from configs import ETH, NBODY, MD17
from tqdm import tqdm
import argparse


# FLAGS
DEBUGGING = True
PROCESS_DATA = True
PROCESS_GRAPHS = True
TRAIN = True
ONLY_LOAD = False

# CHOOSE DATASET
# dataset = "ETH"
# dataset = "NBody"
dataset = "MD17"

if dataset == "ETH":
    config = ETH
elif dataset == "NBody":
    config = NBODY
elif dataset == "MD17":
    config = MD17
config["do_preprocess"] = PROCESS_DATA
config["process_graphs"] = PROCESS_GRAPHS


class ConditionalFlowMatching(L.LightningModule):
    """Conditional Flow Matching model"""

    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        self.dims = 2 if not dataset == "NBody" else 3
        self.window = config["time_window"]
        if config["use_optimal_transport"]:
            self.cfm = ExactOptimalTransportConditionalFlowMatcher(sigma=config["fm_sigma"])
        else:
            self.cfm = ConditionalFlowMatcher(sigma=config["fm_sigma"])
        self.save_hyperparameters()

    def forward(self, t: Tensor, x: Tensor | Data) -> torch.Tensor:
        return self.model(x=x, t=t)

    def training_step(self, batch, batch_idx):
        # Sample location and conditional flow using CFM library
        t, xt_graph, ut, xt = self.model.sample_flow(self.cfm, batch)

        vt = self(t, xt_graph)  # Calculate vector field

        loss = self.model.compute_loss(t, vt, ut, xt_graph, xt)

        self.log("train/loss", loss, on_epoch=True, batch_size=self.config["batch_size"])

        # Plot learning rate
        self.log(
            "train/lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            batch_size=self.config["batch_size"],
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # Sample location and conditional flow using CFM library
        t, xt_graph, ut, xt = self.model.sample_flow(self.cfm, batch)
        # Calculate vector field using model
        vt = self(t, xt_graph)

        val_loss = self.model.compute_loss(t, vt, ut, xt_graph, xt)

        self.log("val/loss", val_loss, on_epoch=True, batch_size=self.config["batch_size"])

        return val_loss

    def on_validation_epoch_end(self):
        # Run validation inference every 10 epochs
        if self.current_epoch % 10 == 0:
            batch = next(iter(self.trainer.datamodule.val_dataloader())).to("cuda")  # type: ignore
            self.model.eval()
            with torch.no_grad():
                rollout, _, _, priors = self.model.generate_batch(self, batch, steps=20)
                ade = compute_ade(rollout, batch, self.config)
                self.log("val/ade", ade, batch_size=self.config["batch_size"])

        if self.current_epoch % 30 == 0:
            val_graph, rollout = batch.clone().cpu(), [x.cpu() for x in rollout]
            plots, metrics = self.model.evaluate(val_graph, priors, rollout, False, DEBUGGING)

            if TRAIN and not DEBUGGING:
                for label, plot in plots.items():
                    label = "val/" + label
                    self.logger.experiment.log({label: wandb.Image(plot)})  # type: ignore
                for label, metric in metrics.items():
                    if label == "ade":
                        continue
                    self.log(f"val/{label}", metric, batch_size=self.config["batch_size"])  # type: ignore

        return None

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            rollouts, all_priors = [], []

            total_test_size = min(config["test_size"], batch.num_graphs)
            batch = Batch.from_data_list(batch.to_data_list()[:total_test_size])
            if config["use_best_of_20"] or config["use_mean_of_5"]:
                idxs = range(total_test_size)
            else:
                idxs = range(0, total_test_size, config["batch_size"])
            for i in tqdm(idxs, desc="Generating test rollouts"):
                if config["use_best_of_20"]:
                    batches = [i for _ in range(20)]
                elif config["use_mean_of_5"]:
                    batches = [i for _ in range(5)]
                else:
                    limit = min(i + config["batch_size"], total_test_size)
                    batches = [j for j in range(i, limit)]
                batches = Batch.from_data_list([batch.get_example(b) for b in batches])
                pred_trajectories, pred, all_trajs, priors = self.model.generate_batch(
                    self, batches, steps=50
                )
                rollouts.extend([pred.cpu() for pred in pred_trajectories])
                all_priors.extend([prior.cpu() for prior in priors])

            plots, metrics = self.model.evaluate(
                batch.cpu(),  # type: ignore
                all_priors,
                rollouts,
                True,
                DEBUGGING,
            )

            if not DEBUGGING:
                for label, plot in plots.items():
                    label = "test/" + label
                    if isinstance(plot, str):
                        self.logger.experiment.log(  # type: ignore
                            {label: wandb.Video(plot, fps=3, format="gif")}
                        )
                    self.logger.experiment.log({label: wandb.Image(plot)})  # type: ignore
                for label, metric in metrics.items():
                    self.log("test/" + label, metric)  # type: ignore
            else:
                print("\nEvaluation metrics:\n")
                for key, value in metrics.items():
                    print(f"{key}: {value}\n")

            return pred_trajectories

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["learning_rate"])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }


def main():
    disable_possible_user_warnings()

    parser = argparse.ArgumentParser()
    for flag, value in config.items():
        if not isinstance(value, dict):
            if isinstance(value, list):
                if len(value) == 0:
                    t = str
                else:
                    t = type(value[0])
                parser.add_argument("--" + flag, type=t, nargs="+")
            else:
                parser.add_argument("--" + flag, type=type(value))

    args = parser.parse_args()
    for arg in vars(args):
        if arg in config and vars(args)[arg] is not None:
            config[arg] = vars(args)[arg]
            print(f"Argument {arg} set to {config[arg]}")
        elif arg == "train" and vars(args)[arg] is not None:
            train = vars(args)[arg]
            print(f"Argument {arg} set to {train}")

    out_channels = 2 if not dataset == "NBody" else 3
    mask_size = int(config["use_inpainting_mask"])
    in_channels = len(config["node_features"].keys()) + mask_size
    flow_model = DynamicGraphNetwork(
        in_channels=in_channels,
        in_edge_channels=len(config["edge_features"].keys()),
        hidden_channels=config["hidden_dim"],
        hidden_edge_channels=config["hidden_dim"],
        out_channels=out_channels,
        config=config,
    )
    if not ONLY_LOAD:
        data_module = GraphsDataModule(config, TRAIN)

    if TRAIN:
        model = ConditionalFlowMatching(flow_model, config).to(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        model = torch.load(config["weights_path"])
        # Overwrite loaded config with config.py
        model.config = config
        flow_model.config = config
        model.model.config = config

    if not DEBUGGING:  # Initialize wandb
        wandb_logger = WandbLogger(
            project="Master_Graduation",
            config=config,
            log_model=True,
            save_dir="logs/wandb",
        )

    if TRAIN:
        run_name = (
            wandb_logger.experiment.name
            if not DEBUGGING and wandb_logger.experiment.name is not None
            else "debug"
        )
        trainer = L.Trainer(
            default_root_dir="logs",
            max_epochs=config["epochs"],
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=[
                ModelCheckpoint(
                    monitor="val/ade",
                    mode="min",
                    dirpath=config["model_path"],
                    save_last=True,
                    filename=f"{time.strftime('%Y-%m-%d')}-{dataset}-{run_name}",
                ),
                RichProgressBar(),
                RichModelSummary(max_depth=4),
            ],
            logger=wandb_logger if not DEBUGGING else None,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            gradient_clip_val=0.5,
        )

        trainer.fit(model, datamodule=data_module)  # type: ignore

        if not DEBUGGING:
            wandb_logger.log_metrics({"train_data_size": len(data_module.train_data)})

        trainer.test(model, datamodule=data_module)  # type: ignore

        if not DEBUGGING:
            wandb_logger.log_metrics({"test_data_size": len(data_module.test_data)})

        name = f"{time.strftime('%Y-%m-%d')}-{dataset}-{config['type']}"
        torch.save(model, f"final_weights/{name}_weights.pth")

    if not TRAIN and not ONLY_LOAD:
        trainer = L.Trainer(
            default_root_dir="logs",
            accelerator="gpu",
            logger=wandb_logger if not DEBUGGING else None,
            num_sanity_val_steps=0,
        )

        trainer.test(model, datamodule=data_module)  # type: ignore

        if not DEBUGGING:
            wandb_logger.log_metrics({"test_data_size": len(data_module.test_data)})

    if not DEBUGGING:
        wandb.finish()

    return model


if __name__ == "__main__":
    main()
