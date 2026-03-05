### Adapted from Metric Flow Matching (https://github.com/kkapusniak/metric-flow-matching)

import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader

from .land import land_metric_tensor
from .rbf import RBFNetwork

class DataManifoldMetric:
    def __init__(
        self,
        args,
        skipped_time_points=None,
        datamodule=None,
    ):
        self.skipped_time_points = skipped_time_points
        self.datamodule = datamodule

        self.gamma = args.gamma_current
        self.rho = args.rho
        self.metric = args.velocity_metric
        self.n_centers = args.n_centers
        self.kappa = args.kappa
        self.metric_epochs = args.metric_epochs
        self.metric_patience = args.metric_patience
        self.lr = args.metric_lr
        self.alpha_metric = args.alpha_metric
        self.image_data = args.data_type == "image"
        self.accelerator = args.accelerator

        self.called_first_time = True
        self.args = args

    def calculate_metric(self, x_t, samples, current_timestep):
        if self.metric == "land":
            M_dd_x_t = (
                land_metric_tensor(x_t, samples, self.gamma, self.rho)
                ** self.alpha_metric
            )
            
        elif self.metric == "rbf":
            if self.called_first_time:
                # Train a single RBF network for all timesteps
                print("Learning single RBF network for all timesteps")
                self.rbf_network = RBFNetwork(
                    current_timestep=0,
                    next_timestep=self.datamodule.num_timesteps - 1,
                    n_centers=self.n_centers,
                    kappa=self.kappa,
                    lr=self.lr,
                    datamodule=self.datamodule,
                    args=self.args
                )
                early_stop_callback = pl.callbacks.EarlyStopping(
                    monitor="MetricModel/train_loss_learn_metric_epoch",
                    patience=self.metric_patience,
                    mode="min",
                )
                trainer = pl.Trainer(
                    max_epochs=self.metric_epochs,
                    accelerator=self.accelerator,
                    logger=WandbLogger(),
                    num_sanity_val_steps=0,
                    callbacks=(
                        [early_stop_callback] if not self.image_data else None
                    ),
                )
                if self.image_data:
                    self.dataloader = DataLoader(
                        self.datamodule.all_data,
                        batch_size=128,
                        shuffle=True,
                    )
                    trainer.fit(self.rbf_network, self.dataloader)
                else:
                    trainer.fit(self.rbf_network, self.datamodule)
                self.called_first_time = False
                print("Learning RBF network... Done")
            M_dd_x_t = self.rbf_network.compute_metric(
                x_t,
                epsilon=self.rho,
                alpha=self.alpha_metric,
                image_hx=self.image_data,
            )
        return M_dd_x_t

    def calculate_velocity(self, x_t, u_t, samples, timestep):

        if len(u_t.shape) > 2:
            u_t = u_t.reshape(u_t.shape[0], -1)
            x_t = x_t.reshape(x_t.shape[0], -1)
        M_dd_x_t = self.calculate_metric(x_t, samples, timestep).to(u_t.device)

        # Clamp to prevent NaN from sqrt of negative values when the RBF
        # metric tensor is not positive-definite for some inputs
        velocity = torch.sqrt(torch.clamp(((u_t**2) * M_dd_x_t).sum(dim=-1), min=0))
        ut_sum = (u_t**2).sum(dim=-1)
        metric_sum = M_dd_x_t.sum(dim=-1)
        return velocity, ut_sum, metric_sum
