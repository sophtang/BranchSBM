import sys
import yaml
import string
import secrets
import os
import torch
import wandb
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torchdyn.core import NeuralODE
from src.utils import plot_images_trajectory
from src.networks.utils import flow_model_torch_wrapper


def load_config(path):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def merge_config(args, config_updates):
    for key, value in config_updates.items():
        if not hasattr(args, key):
            raise ValueError(
                f"Unknown configuration parameter '{key}' found in the config file."
            )
        setattr(args, key, value)
    return args

def generate_group_string(length=16):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def dataset_name2datapath(dataset_name, working_dir):
    if dataset_name in ["lidar", "lidarsingle"]:
        return os.path.join(working_dir, "data", "rainier2-thin.las")
    elif dataset_name in ["mouse", "mousesingle"]:
        return os.path.join(working_dir, "data", "mouse_hematopoiesis.csv")
    elif dataset_name in ["clonidine50D", "clonidine100D", "clonidine150D", "clonidine50Dsingle", "clonidine100Dsingle", "clonidine150Dsingle"]:
        return os.path.join(working_dir, "data", "pca_and_leiden_labels.csv")
    elif dataset_name in ["trametinib", "trametinibsingle"]:
        return os.path.join(working_dir, "data", "Trametinib_5.0uM_pca_and_leidenumap_labels.csv")
    elif dataset_name in ["veres", "veressingle"]:
        return os.path.join(working_dir, "data", "Veres_alltime.csv")
    else:
        raise ValueError("Dataset not recognized")


def create_callbacks(args, phase, data_type, run_id, datamodule=None):

    dirpath = os.path.join(
        args.working_dir,
        "checkpoints",
        data_type,
        str(args.run_name),
        f"{phase}_model",
    )

    if phase == "geopath":
        early_stop_callback = EarlyStopping(
            monitor="BranchPathNet/train_loss_geopath_epoch",
            patience=args.patience_geopath,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="BranchPathNet/train_loss_geopath_epoch",
            mode="min",
            save_top_k=1,
        )
        callbacks = [checkpoint_callback, early_stop_callback]
    elif phase == "flow":
        early_stop_callback = EarlyStopping(
            monitor="FlowNet/train_loss_cfm",
            patience=args.patience,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="FlowNet/train_loss_cfm",
            mode="min",
            save_top_k=1,
        )
        callbacks = [checkpoint_callback, early_stop_callback]
    elif phase == "growth":
        early_stop_callback = EarlyStopping(
            monitor="GrowthNet/train_loss",
            patience=args.patience,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="GrowthNet/train_loss",
            mode="min",
            save_top_k=1,
        )
        callbacks = [checkpoint_callback, early_stop_callback]
    elif phase == "joint":
        early_stop_callback = EarlyStopping(
            monitor="JointTrain/train_loss",
            patience=args.patience,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            mode="min",
            save_top_k=1,
        )
        callbacks = [checkpoint_callback, early_stop_callback]
    else:
        raise ValueError("Unknown phase")
    return callbacks


class PlottingCallback(Callback):
    def __init__(self, plot_interval, datamodule):
        self.plot_interval = plot_interval
        self.datamodule = datamodule

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        pl_module.flow_net.train(mode=False)
        if epoch % self.plot_interval == 0 and epoch != 0:
            node = NeuralODE(
                flow_model_torch_wrapper(pl_module.flow_net).to(self.datamodule.device),
                solver="tsit5",
                sensitivity="adjoint",
                atol=1e-5,
                rtol=1e-5,
            )

            for mode in ["train", "val"]:
                x0 = getattr(self.datamodule, f"{mode}_x0")
                x0 = x0[0:15]
                fig = self.trajectory_and_plot(x0, node, self.datamodule)
                wandb.log({f"Trajectories {mode.capitalize()}": wandb.Image(fig)})
        pl_module.flow_net.train(mode=True)

    def trajectory_and_plot(self, x0, node, datamodule):
        selected_images = x0[0:15]
        with torch.no_grad():
            traj = node.trajectory(
                selected_images.to(datamodule.device),
                t_span=torch.linspace(0, 1, 100).to(datamodule.device),
            )

        traj = traj.transpose(0, 1)
        traj = traj.reshape(*traj.shape[0:2], *datamodule.dim)

        fig = plot_images_trajectory(
            traj.to(datamodule.device),
            datamodule.vae.to(datamodule.device),
            datamodule.process,
            num_steps=5,
        )
        return fig