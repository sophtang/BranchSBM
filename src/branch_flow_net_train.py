import os
import sys
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
from .networks.utils import flow_model_torch_wrapper
from .utils import wasserstein, plot_lidar
from .ema import EMA

class BranchFlowNetTrainBase(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        flow_nets,
        skipped_time_points=None,
        ot_sampler=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        
        self.flow_matcher = flow_matcher
        self.flow_nets = flow_nets # list of flow networks for each branch
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points

        self.optimizer_name = args.flow_optimizer
        self.lr = args.flow_lr
        self.weight_decay = args.flow_weight_decay
        self.whiten = args.whiten
        self.working_dir = args.working_dir
        
        #branching 
        self.branches = len(flow_nets)

    def forward(self, t, xt, branch_idx):
        # output velocity given branch_idx
        return self.flow_nets[branch_idx](t, xt)

    def _compute_loss(self, main_batch):
        
        x0s = [main_batch["x0"][0]]
        w0s = [main_batch["x0"][1]]
        
        x1s_list = []
        w1s_list = [] 
        
        if self.branches > 1:
            for i in range(self.branches):
                x1s_list.append([main_batch[f"x1_{i+1}"][0]])
                w1s_list.append([main_batch[f"x1_{i+1}"][1]])
        else:
            x1s_list.append([main_batch["x1"][0]])
            w1s_list.append([main_batch["x1"][1]])
        
        assert len(x1s_list) == self.branches, "Mismatch between x1s_list and expected branches"
        
        loss = 0
        for branch_idx in range(self.branches):
            ts, xts, uts = self._process_flow(x0s, x1s_list[branch_idx], branch_idx)

            t = torch.cat(ts)
            xt = torch.cat(xts)
            ut = torch.cat(uts)
            vt = self(t[:, None], xt, branch_idx)

            loss += mean_squared_error(vt, ut)

        return loss

    def _process_flow(self, x0s, x1s, branch_idx):
        ts, xts, uts = [], [], []
        t_start = self.timesteps[0]

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            if self.ot_sampler is not None:
                x0, x1 = self.ot_sampler.sample_plan(
                    x0,
                    x1,
                    replace=True,
                )
            if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                t_start_next = self.timesteps[i + 2]
            else:
                t_start_next = self.timesteps[i + 1]
            
            # edit to sample from correct flow matcher
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next, branch_idx
            )

            ts.append(t)

            xts.append(xt)
            uts.append(ut)
            t_start = t_start_next
        return ts, xts, uts

    def training_step(self, batch, batch_idx):
        # Handle both dict and tuple batch formats from CombinedLoader
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if isinstance(batch, dict) and "train_samples" in batch:
            main_batch = batch["train_samples"]
            if isinstance(main_batch, tuple):
                main_batch = main_batch[0]
        else:
            # Fallback
            main_batch = batch.get("train_samples", batch)
            
        print("Main batch length")
        print(len(main_batch["x0"]))
        
        # edited to simulate 100 steps
        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch["x0"])).tolist()
        loss = self._compute_loss(main_batch)
        if self.flow_matcher.alpha != 0:
            self.log(
                "FlowNet/mean_geopath_cfm",
                (self.flow_matcher.geopath_net_output.abs().mean()),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.log(
            "FlowNet/train_loss_cfm",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Handle both dict and tuple batch formats from CombinedLoader
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if isinstance(batch, dict) and "val_samples" in batch:
            main_batch = batch["val_samples"]
            if isinstance(main_batch, tuple):
                main_batch = main_batch[0]
        else:
            # Fallback
            main_batch = batch.get("val_samples", batch)
            
        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch["x0"])).tolist()
        val_loss = self._compute_loss(main_batch)
        self.log(
            "FlowNet/val_loss_cfm",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return val_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        
        for net in self.flow_nets:
            if isinstance(net, EMA):
                net.update_ema()

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
            )

        return optimizer


class FlowNetTrainTrajectory(BranchFlowNetTrainBase):
    def test_step(self, batch, batch_idx):
        data_type = self.args.data_type
        node = NeuralODE(
            flow_model_torch_wrapper(self.flow_nets),
            solver="euler",
            sensitivity="adjoint",
            atol=1e-5,
            rtol=1e-5,
        )

        t_exclude = self.skipped_time_points[0] if self.skipped_time_points else None
        if t_exclude is not None:
            traj = node.trajectory(
                batch[t_exclude - 1],
                t_span=torch.linspace(
                    self.timesteps[t_exclude - 1], self.timesteps[t_exclude], 101
                ),
            )
            X_mid_pred = traj[-1]
            traj = node.trajectory(
                batch[t_exclude - 1],
                t_span=torch.linspace(
                    self.timesteps[t_exclude - 1],
                    self.timesteps[t_exclude + 1],
                    101,
                ),
            )
            
            EMD = wasserstein(X_mid_pred, batch[t_exclude], p=1)
            self.final_EMD = EMD

            self.log("test_EMD", EMD, on_step=False, on_epoch=True, prog_bar=True)

class FlowNetTrainCell(BranchFlowNetTrainBase):
    def test_step(self, batch, batch_idx):
        x0 = batch[0]["test_samples"][0]["x0"][0]  # [B, D]
        dataset_points = batch[0]["test_samples"][0]["dataset"][0]  # full dataset, [N, D]
        t_span = torch.linspace(0, 1, 101)

        all_trajs = []

        for i, flow_net in enumerate(self.flow_nets):
            node = NeuralODE(
                flow_model_torch_wrapper(flow_net),
                solver="euler",
                sensitivity="adjoint",
            )

            with torch.no_grad():
                traj = node.trajectory(x0, t_span).cpu()  # [T, B, D]

            if self.whiten:
                traj_shape = traj.shape
                traj = traj.reshape(-1, traj.shape[-1])
                traj = self.trainer.datamodule.scaler.inverse_transform(
                    traj.cpu().detach().numpy()
                ).reshape(traj_shape)
                dataset_points = self.trainer.datamodule.scaler.inverse_transform(
                    dataset_points.cpu().detach().numpy()
                )

            traj = torch.tensor(traj)
            traj = torch.transpose(traj, 0, 1)  # [B, T, D]
            all_trajs.append(traj)

        dataset_2d = dataset_points[:, :2] if isinstance(dataset_points, torch.Tensor) else dataset_points[:, :2]

        # ===== Plot all 2D trajectories together with dataset and start/end points =====
        fig, ax = plt.subplots(figsize=(6, 5))
        dataset_2d = dataset_2d.cpu().numpy()
        ax.scatter(dataset_2d[:, 0], dataset_2d[:, 1], c="gray", s=1, alpha=0.5, label="Dataset", zorder=1)
        for traj in all_trajs:
            traj_2d = traj[..., :2]  # [B, T, 2]
            for i in range(traj_2d.shape[0]):
                ax.plot(traj_2d[i, :, 0], traj_2d[i, :, 1], alpha=0.8, zorder=2)
                ax.scatter(traj_2d[i, 0, 0], traj_2d[i, 0, 1], c='green', s=10, label="t=0" if i == 0 else "", zorder=3)
                ax.scatter(traj_2d[i, -1, 0], traj_2d[i, -1, 1], c='red', s=10, label="t=1" if i == 0 else "", zorder=3)

        ax.set_title("All Branch Trajectories (2D) with Dataset")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.axis("equal")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
            
        run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
        results_dir = os.path.join(self.args.working_dir, 'results', run_name)
        save_path = os.path.join(results_dir, 'figures')
        
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{self.args.data_name}_all_branches.png', dpi=300)
        plt.close()

        # ===== Plot each 2D trajectory separately with dataset and endpoints =====
        for i, traj in enumerate(all_trajs):
            traj_2d = traj[..., :2]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(dataset_2d[:, 0], dataset_2d[:, 1], c="gray", s=1, alpha=0.5, label="Dataset", zorder=1)
            for j in range(traj_2d.shape[0]):
                ax.plot(traj_2d[j, :, 0], traj_2d[j, :, 1], alpha=0.9, zorder=2)
                ax.scatter(traj_2d[j, 0, 0], traj_2d[j, 0, 1], c='green', s=12, label="t=0" if j == 0 else "", zorder=3)
                ax.scatter(traj_2d[j, -1, 0], traj_2d[j, -1, 1], c='red', s=12, label="t=1" if j == 0 else "", zorder=3)

            ax.set_title(f"Branch {i + 1} Trajectories (2D) with Dataset")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.axis("equal")
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend()
            plt.savefig(f'{save_path}/{self.args.data_name}_branch_{i + 1}.png', dpi=300)
            plt.close()

class FlowNetTrainLidar(BranchFlowNetTrainBase):
    def test_step(self, batch, batch_idx):
        # Handle both tuple and dict batch formats from CombinedLoader
        if isinstance(batch, dict):
            main_batch = batch["test_samples"][0]
            metric_batch = batch["metric_samples"][0]
        else:
            # batch is a tuple: (test_samples, metric_samples)
            main_batch = batch[0][0]
            metric_batch = batch[1][0]
                
        x0 = main_batch["x0"][0] # [B, D]
        cloud_points = main_batch["dataset"][0]  # full dataset, [N, D]
        t_span = torch.linspace(0, 1, 101)

        all_trajs = []

        for i, flow_net in enumerate(self.flow_nets):
            node = NeuralODE(
                flow_model_torch_wrapper(flow_net),
                solver="euler",
                sensitivity="adjoint",
            )

            with torch.no_grad():
                traj = node.trajectory(x0, t_span).cpu()  # [T, B, D]

            if self.whiten:
                traj_shape = traj.shape
                traj = traj.reshape(-1, 3)
                traj = self.trainer.datamodule.scaler.inverse_transform(
                    traj.cpu().detach().numpy()
                ).reshape(traj_shape)

            traj = torch.tensor(traj)
            traj = torch.transpose(traj, 0, 1)  # [B, T, D]
            all_trajs.append(traj)

        # Inverse-transform the point cloud once
        if self.whiten:
            cloud_points = torch.tensor(
                self.trainer.datamodule.scaler.inverse_transform(
                    cloud_points.cpu().detach().numpy()
                )
            )

        # Create directory for saving figures
        run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
        results_dir = os.path.join(self.args.working_dir, 'results', run_name)
        lidar_fig_dir = os.path.join(results_dir, 'figures')
        os.makedirs(lidar_fig_dir, exist_ok=True)

        # ===== Plot all trajectories together =====
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.view_init(elev=30, azim=-115, roll=0)
        for i, traj in enumerate(all_trajs):
            plot_lidar(ax, cloud_points, xs=traj, branch_idx=i)
        plt.savefig(os.path.join(lidar_fig_dir, 'lidar_all_branches.png'), dpi=300)
        plt.close()

        # ===== Plot each trajectory separately =====
        for i, traj in enumerate(all_trajs):
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
            ax.view_init(elev=30, azim=-115, roll=0)
            plot_lidar(ax, cloud_points, xs=traj, branch_idx=i)
            plt.savefig(os.path.join(lidar_fig_dir, f'lidar_branch_{i + 1}.png'), dpi=300)
            plt.close()