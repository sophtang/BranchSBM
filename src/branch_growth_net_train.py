import os
import sys
import torch
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import AdamW
from torchmetrics.functional import mean_squared_error
from torchdyn.core import NeuralODE
import numpy as np
import lpips
from .networks.utils import flow_model_torch_wrapper
from .utils import plot_lidar
from .ema import EMA
from torchdiffeq import odeint as odeint2
from .losses.energy_loss import EnergySolver, ReconsLoss

class GrowthNetTrain(pl.LightningModule):
    def __init__(
        self,
        flow_nets,
        growth_nets,
        skipped_time_points=None,
        ot_sampler=None,
        args=None,
        
        state_cost=None,
        data_manifold_metric=None,
        
        joint = False
    ):
        super().__init__()
        #self.save_hyperparameters()
        self.flow_nets = flow_nets
        
        if not joint:
            for param in self.flow_nets.parameters():
                param.requires_grad = False
        
        self.growth_nets = growth_nets # list of growth networks for each branch
        
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points

        self.optimizer_name = args.growth_optimizer
        self.lr = args.growth_lr
        self.weight_decay = args.growth_weight_decay
        self.whiten = args.whiten
        self.working_dir = args.working_dir
        
        self.args = args
        
        #branching 
        self.state_cost = state_cost
        self.data_manifold_metric = data_manifold_metric
        self.branches = len(growth_nets)
        self.metric_clusters = args.metric_clusters
        
        self.recons_loss = ReconsLoss()
        
        # loss weights
        self.lambda_energy = args.lambda_energy
        self.lambda_mass = args.lambda_mass
        self.lambda_match = args.lambda_match
        self.lambda_recons = args.lambda_recons
        
        self.joint = joint

    def forward(self, t, xt, branch_idx):
        # output growth rate given branch_idx
        return self.growth_nets[branch_idx](t, xt)

    def _compute_loss(self, main_batch,  metric_samples_batch=None, validation=False):
        x0s = main_batch["x0"][0]
        w0s = main_batch["x0"][1]
        x1s_list = []
        w1s_list = [] 
        
        if self.branches > 1:
            for i in range(self.branches):
                x1s_list.append([main_batch[f"x1_{i+1}"][0]])
                w1s_list.append([main_batch[f"x1_{i+1}"][1]])
        else:
            x1s_list.append([main_batch["x1"][0]])
            w1s_list.append([main_batch["x1"][1]])
        
        if self.args.manifold:
            #changed
            if self.metric_clusters == 7 and self.branches == 6:
                # Weinreb 6-branch scenario: cluster 0 (root) → clusters 1-6 (6 branches)
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2 (branch 2)
                    (metric_samples_batch[0], metric_samples_batch[3]),  # x0 → x1_3 (branch 3)
                    (metric_samples_batch[0], metric_samples_batch[4]),  # x0 → x1_4 (branch 4)
                    (metric_samples_batch[0], metric_samples_batch[5]),  # x0 → x1_5 (branch 5)
                    (metric_samples_batch[0], metric_samples_batch[6]),  # x0 → x1_6 (branch 6)
                ]
            elif self.metric_clusters == 4:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2 (branch 2)
                    (metric_samples_batch[0], metric_samples_batch[3]),
                ]
            elif self.metric_clusters == 3:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2 (branch 2)
                ]
            elif self.metric_clusters == 2 and self.branches == 2:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_2 (branch 2)
                ]
            elif self.metric_clusters == 2:
                # For any number of branches with 2 metric clusters (initial vs remaining)
                # All branches use the same metric cluster pair
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1])  # x0 → all branches
                ] * self.branches
            else:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                ]
        
        batch_size = x0s.shape[0]
        
        assert len(x1s_list) == self.branches, "Mismatch between x1s_list and expected branches"
        
        energy_loss = [0.] * self.branches
        mass_loss = 0.
        neg_weight_penalty = 0.
        match_loss = [0.] * self.branches
        recons_loss = [0.] * self.branches
        
        dtype = x0s[0].dtype
        #w0s = torch.zeros((batch_size, 1), dtype=dtype) 
        m0s = torch.zeros_like(w0s, dtype=dtype)
        start_state = (x0s, w0s, m0s)
        
        xt = [x0s.clone() for _ in range(self.branches)]
        w0_branch = torch.zeros_like(w0s, dtype=dtype)
        w0_branches = []
        w0_branches.append(w0s)
        for _ in range(self.branches - 1):
            w0_branches.append(w0_branch)
        #w0_branches = [w0_branch.clone() for _ in range(self.branches - 1)]
        wt = w0_branches
        
        mt = [m0s.clone() for _ in range(self.branches)]
        
        # loop through timesteps
        for step_idx, (s, t) in enumerate(zip(self.timesteps[:-1], self.timesteps[1:])):
            time = torch.Tensor([s, t])
            
            total_w_t = 0
            # loop through branches
            for i in range(self.branches):
                
                if self.args.manifold:
                    start_samples, end_samples = branch_sample_pairs[i]
                    samples = torch.cat([start_samples, end_samples], dim=0)
                else:
                    samples = None
                    
                # initialize weight and energy
                start_state = (xt[i], wt[i], mt[i])
                
                # loop over timesteps
                xt_next, wt_next, mt_next = self.take_step(time, start_state, i, samples, timestep_idx=step_idx)
                
                # placeholders for next state
                xt_last = xt_next[-1]
                wt_last = wt_next[-1]
                mt_last = mt_next[-1]
                
                total_w_t += wt_last

                energy_loss[i] += (mt_last - mt[i])
                neg_weight_penalty += torch.relu(-wt_last).sum()
                
                # update branch state
                xt[i] = xt_last.clone().detach()
                wt[i] = wt_last.clone().detach()
                mt[i] = mt_last.clone().detach()

            # calculate mass loss from all branches
            target = torch.ones_like(total_w_t)
            mass_loss += mean_squared_error(total_w_t, target)
        
        # calculate loss that matches final weights 
        for i in range(self.branches):
            match_loss[i] = mean_squared_error(wt[i], w1s_list[i][0])
            # compute reconstruction loss
            recons_loss[i] = self.recons_loss(xt[i], x1s_list[i][0])
        
        # average across time steps (loop runs len(timesteps)-1 times)
        mass_loss = mass_loss / max(len(self.timesteps) - 1, 1)
        
        
        # Weighted mean across branches (inversely weighted by cluster size)
        # Get cluster sizes from datamodule if available
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'cluster_sizes'):
            cluster_sizes = self.trainer.datamodule.cluster_sizes
            max_size = max(cluster_sizes)
            # Inverse weighting: smaller clusters get higher weight
            branch_weights = torch.tensor([max_size / size for size in cluster_sizes], 
                                         dtype=energy_loss[0].dtype, device=energy_loss[0].device)
            # Normalize weights to sum to num_branches for fair comparison
            branch_weights = branch_weights * self.branches / branch_weights.sum()
            
            energy_loss = torch.mean(torch.stack([e.mean() for e in energy_loss]) * branch_weights)
            match_loss = torch.mean(torch.stack(match_loss) * branch_weights)
            recons_loss = torch.mean(torch.stack(recons_loss) * branch_weights)
        else:
            # Fallback to uniform weighting
            energy_loss = torch.mean(torch.stack([e.mean() for e in energy_loss]))
            match_loss = torch.mean(torch.stack(match_loss))
            recons_loss = torch.mean(torch.stack(recons_loss))
        
        loss = (self.lambda_energy * energy_loss) + (self.lambda_mass * (mass_loss + neg_weight_penalty)) + (self.lambda_match * match_loss) \
            + (self.lambda_recons * recons_loss)
            
        if self.joint:
            if validation:
                self.log("JointTrain/val_mass_loss", mass_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/val_neg_penalty_loss", neg_weight_penalty, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/val_match_loss", match_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/val_energy_loss", energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/val_recons_loss", recons_loss, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log("JointTrain/train_mass_loss", mass_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/train_neg_penalty_loss", neg_weight_penalty, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/train_match_loss", match_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/train_energy_loss", energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/train_recons_loss", recons_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            if validation:
                self.log("GrowthNet/val_mass_loss", mass_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/val_neg_penalty_loss", neg_weight_penalty, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/val_match_loss", match_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/val_energy_loss", energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/val_recons_loss", recons_loss, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log("GrowthNet/train_mass_loss", mass_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/train_neg_penalty_loss", neg_weight_penalty, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/train_match_loss", match_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/train_energy_loss", energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/train_recons_loss", recons_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def take_step(self, t, start_state, branch_idx, samples=None, timestep_idx=0):
        
        flow_net = self.flow_nets[branch_idx]
        growth_net = self.growth_nets[branch_idx]
        
        
        x_t, w_t, m_t = odeint2(EnergySolver(flow_net, growth_net, self.state_cost, self.data_manifold_metric, samples, timestep_idx), start_state, t, options=dict(step_size=0.1),method='euler')
        
        return x_t, w_t, m_t
    
    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if isinstance(batch, dict) and "train_samples" in batch:
            main_batch = batch["train_samples"]
            metric_batch = batch["metric_samples"]
            if isinstance(main_batch, tuple):
                main_batch = main_batch[0]
            if isinstance(metric_batch, tuple):
                metric_batch = metric_batch[0]
        else:
            # Fallback
            main_batch = batch.get("train_samples", batch)
            metric_batch = batch.get("metric_samples", [])
        
        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch["x0"])).tolist()
        loss = self._compute_loss(main_batch, metric_batch, validation=False)
        
        if self.joint:
            self.log(
                "JointTrain/train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                "GrowthNet/train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if isinstance(batch, dict) and "val_samples" in batch:
            main_batch = batch["val_samples"]
            metric_batch = batch["metric_samples"]
            if isinstance(main_batch, tuple):
                main_batch = main_batch[0]
            if isinstance(metric_batch, tuple):
                metric_batch = metric_batch[0]
        else:
            # Fallback
            main_batch = batch.get("val_samples", batch)
            metric_batch = batch.get("metric_samples", [])

        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch["x0"])).tolist()
        val_loss = self._compute_loss(main_batch, metric_batch, validation=True)
        
        if self.joint:
            self.log(
                "JointTrain/val_loss",
                val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                "GrowthNet/val_loss",
                val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return val_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        for net in self.growth_nets:
            if isinstance(net, EMA):
                net.update_ema()
        if self.joint:
            for net in self.flow_nets:
                if isinstance(net, EMA):
                    net.update_ema()

    def configure_optimizers(self):
        params = []
        for net in self.growth_nets:
            params += list(net.parameters())
            
        if self.joint:
            for net in self.flow_nets:
                params += list(net.parameters())
        
        if self.optimizer_name == "adamw":
            optimizer = AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.lr,
            )

        return optimizer
    
    @torch.no_grad()
    def get_mass_and_position(self, main_batch, metric_samples_batch=None):
        if isinstance(main_batch, dict):
            main_batch = main_batch
        else:
            main_batch = main_batch[0]
            
        x0s = main_batch["x0"][0]
        w0s = main_batch["x0"][1]

        if self.args.manifold:
            if self.metric_clusters == 4:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2 (branch 2)
                    (metric_samples_batch[0], metric_samples_batch[3]),
                ]
            elif self.metric_clusters == 3:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2 (branch 2)
                ]
            elif self.metric_clusters == 2 and self.branches == 2:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_2 (branch 2)
                ]
            elif self.metric_clusters == 2:
                # For any number of branches with 2 metric clusters (initial vs remaining)
                # All branches use the same metric cluster pair
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1])  # x0 → all branches
                ] * self.branches
            else:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                ]

        batch_size = x0s.shape[0]
        dtype = x0s[0].dtype

        m0s = torch.zeros_like(w0s, dtype=dtype)
        xt = [x0s.clone() for _ in range(self.branches)]

        w0_branch = torch.zeros_like(w0s, dtype=dtype)
        w0_branches = []
        w0_branches.append(w0s)
        for _ in range(self.branches - 1):
            w0_branches.append(w0_branch)

        wt = w0_branches
        mt = [m0s.clone() for _ in range(self.branches)]

        time_points = []
        mass_over_time = [[] for _ in range(self.branches)]
        energy_over_time = [[] for _ in range(self.branches)]
        # record per-sample weights at each time for each branch (to allow OT with per-sample masses)
        weights_over_time = [[] for _ in range(self.branches)]
        all_trajs = [[] for _ in range(self.branches)]

        t_span = torch.linspace(0, 1, 101)
        for step_idx, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            time_points.append(t.item())
            time = torch.Tensor([s, t])

            for i in range(self.branches):
                if self.args.manifold:
                    start_samples, end_samples = branch_sample_pairs[i]
                    samples = torch.cat([start_samples, end_samples], dim=0)
                else:
                    samples = None

                start_state = (xt[i], wt[i], mt[i])
                xt_next, wt_next, mt_next = self.take_step(time, start_state, i, samples, timestep_idx=step_idx)

                xt[i] = xt_next[-1].clone().detach()
                wt[i] = wt_next[-1].clone().detach()
                mt[i] = mt_next[-1].clone().detach()

                all_trajs[i].append(xt[i].clone().detach())
                mass_over_time[i].append(wt[i].mean().item())
                energy_over_time[i].append(mt[i].mean().item())
                # store per-sample weights (clone to detach from graph)
                try:
                    weights_over_time[i].append(wt[i].clone().detach())
                except Exception:
                    # fallback: store mean as singleton tensor
                    weights_over_time[i].append(torch.tensor(wt[i].mean().item()).unsqueeze(0))
                
        return time_points, xt, all_trajs, mass_over_time, energy_over_time, weights_over_time

    @torch.no_grad()
    def _plot_mass_and_energy(self, main_batch, metric_samples_batch=None, save_dir=None):
        x0s = main_batch["x0"][0]
        w0s = main_batch["x0"][1]

        if self.args.manifold:
            if self.metric_clusters == 7 and self.branches == 6:
                # Weinreb 6-branch scenario: cluster 0 (root) → clusters 1-6 (6 branches)
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2 (branch 2)
                    (metric_samples_batch[0], metric_samples_batch[3]),  # x0 → x1_3 (branch 3)
                    (metric_samples_batch[0], metric_samples_batch[4]),  # x0 → x1_4 (branch 4)
                    (metric_samples_batch[0], metric_samples_batch[5]),  # x0 → x1_5 (branch 5)
                    (metric_samples_batch[0], metric_samples_batch[6]),  # x0 → x1_6 (branch 6)
                ]
            elif self.metric_clusters == 4:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2 (branch 2)
                    (metric_samples_batch[0], metric_samples_batch[3]),
                ]
            elif self.metric_clusters == 3:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2 (branch 2)
                ]
            elif self.metric_clusters == 2 and self.branches == 2:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_2 (branch 2)
                ]
            else:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1 (branch 1)
                ]

        batch_size = x0s.shape[0]
        dtype = x0s[0].dtype

        m0s = torch.zeros_like(w0s, dtype=dtype)
        xt = [x0s.clone() for _ in range(self.branches)]

        w0_branch = torch.zeros_like(w0s, dtype=dtype)
        w0_branches = []
        w0_branches.append(w0s)
        for _ in range(self.branches - 1):
            w0_branches.append(w0_branch)

        wt = w0_branches
        mt = [m0s.clone() for _ in range(self.branches)]

        time_points = []
        mass_over_time = [[] for _ in range(self.branches)]
        energy_over_time = [[] for _ in range(self.branches)]

        t_span = torch.linspace(0, 1, 101)
        for step_idx, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            time_points.append(t.item())
            time = torch.Tensor([s, t])

            for i in range(self.branches):
                if self.args.manifold:
                    start_samples, end_samples = branch_sample_pairs[i]
                    samples = torch.cat([start_samples, end_samples], dim=0)
                else:
                    samples = None

                start_state = (xt[i], wt[i], mt[i])
                xt_next, wt_next, mt_next = self.take_step(time, start_state, i, samples, timestep_idx=step_idx)

                xt[i] = xt_next[-1].clone().detach()
                wt[i] = wt_next[-1].clone().detach()
                mt[i] = mt_next[-1].clone().detach()

                mass_over_time[i].append(wt[i].mean().item())
                energy_over_time[i].append(mt[i].mean().item())

        if save_dir is None:
            run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
            save_dir = os.path.join(self.args.working_dir, 'results', run_name, 'figures')
        os.makedirs(save_dir, exist_ok=True)

        # Use tab10 colormap to get visually distinct colors
        if self.args.branches == 3:
            branch_colors = ['#9793F8', '#50B2D7', '#D577FF']  # tuple of RGBs
        else:
            branch_colors = ['#50B2D7', '#D577FF']  # tuple of RGBs

        # --- Plot Mass ---
        plt.figure(figsize=(8, 5))
        for i in range(self.branches):
            color = branch_colors[i]
            plt.plot(time_points, mass_over_time[i], color=color, linewidth=2.5, label=f"Mass Branch {i}")
        plt.xlabel("Time")
        plt.ylabel("Mass")
        plt.title("Mass Evolution per Branch")
        plt.legend()
        plt.grid(True)
        if self.joint:
            mass_path = os.path.join(save_dir, f"{self.args.data_name}_joint_mass.png")
        else:
            mass_path = os.path.join(save_dir, f"{self.args.data_name}_growth_mass.png")
        plt.savefig(mass_path, dpi=300, bbox_inches="tight")
        plt.close()

        # --- Plot Energy ---
        plt.figure(figsize=(8, 5))
        for i in range(self.branches):
            color = branch_colors[i]
            plt.plot(time_points, energy_over_time[i], color=color, linewidth=2.5, label=f"Energy Branch {i}")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.title("Energy Evolution per Branch")
        plt.legend()
        plt.grid(True)
        if self.joint:
            energy_path = os.path.join(save_dir, f"{self.args.data_name}_joint_energy.png")
        else: 
            energy_path = os.path.join(save_dir, f"{self.args.data_name}_growth_energy.png")
        plt.savefig(energy_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        
class GrowthNetTrainLidar(GrowthNetTrain):
    def test_step(self, batch, batch_idx):
        # Handle both tuple and dict batch formats from CombinedLoader
        if isinstance(batch, dict):
            main_batch = batch["test_samples"][0]
            metric_batch = batch["metric_samples"][0]
        else:
            # batch is a tuple: (test_samples, metric_samples)
            main_batch = batch[0][0]
            metric_batch = batch[1][0]
        
        self._plot_mass_and_energy(main_batch, metric_batch)
        
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

        # ===== Plot all trajectories together =====
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.view_init(elev=30, azim=-115, roll=0)
        for i, traj in enumerate(all_trajs):
            plot_lidar(ax, cloud_points, xs=traj, branch_idx=i)
        run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
        results_dir = os.path.join(self.args.working_dir, 'results', run_name)
        lidar_fig_dir = os.path.join(results_dir, 'figures')
        os.makedirs(lidar_fig_dir, exist_ok=True)
        if self.joint:
            plt.savefig(os.path.join(lidar_fig_dir, 'joint_lidar_all_branches.png'), dpi=300)
        else:
            plt.savefig(os.path.join(lidar_fig_dir, 'growth_lidar_all_branches.png'), dpi=300)
        plt.close()

        # ===== Plot each trajectory separately =====
        for i, traj in enumerate(all_trajs):
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
            ax.view_init(elev=30, azim=-115, roll=0)
            plot_lidar(ax, cloud_points, xs=traj, branch_idx=i)
            if self.joint:
                plt.savefig(os.path.join(lidar_fig_dir, f'joint_lidar_branch_{i + 1}.png'), dpi=300)
            else:
                plt.savefig(os.path.join(lidar_fig_dir, f'growth_lidar_branch_{i + 1}.png'), dpi=300)
            plt.close()
        
class GrowthNetTrainCell(GrowthNetTrain):
    def test_step(self, batch, batch_idx):
        if self.args.data_type in ["scrna", "tahoe"]:
            main_batch = batch[0]["test_samples"][0]
            metric_batch = batch[0]["metric_samples"][0]
        else: 
            main_batch = batch["test_samples"][0]
            metric_batch = batch["metric_samples"][0]
        
        self._plot_mass_and_energy(main_batch, metric_batch)


class SequentialGrowthNetTrain(pl.LightningModule):
    """
    Sequential growth network training for multi-timepoint data.
    Learns growth rates for transitions between consecutive timepoints.
    """
    def __init__(
        self,
        flow_nets,
        growth_nets,
        skipped_time_points=None,
        ot_sampler=None,
        args=None,
        data_manifold_metric=None,
        joint=False
    ):
        super().__init__()
        self.flow_nets = flow_nets
        
        if not joint:
            for param in self.flow_nets.parameters():
                param.requires_grad = False
        
        self.growth_nets = growth_nets
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points

        self.optimizer_name = args.growth_optimizer
        self.lr = args.growth_lr
        self.weight_decay = args.growth_weight_decay
        self.whiten = args.whiten
        self.working_dir = args.working_dir
        
        self.args = args
        self.data_manifold_metric = data_manifold_metric
        self.branches = len(growth_nets)
        self.metric_clusters = args.metric_clusters
        
        self.recons_loss = ReconsLoss()
        
        # loss weights
        self.lambda_energy = args.lambda_energy
        self.lambda_mass = args.lambda_mass
        self.lambda_match = args.lambda_match
        self.lambda_recons = args.lambda_recons
        
        self.joint = joint
        self.num_timepoints = None
        self.timepoint_keys = None

    def forward(self, t, xt, branch_idx):
        return self.growth_nets[branch_idx](t, xt)

    def setup(self, stage=None):
        """Initialize timepoint keys before training/validation starts."""
        if self.timepoint_keys is None:
            timepoint_data = self.trainer.datamodule.get_timepoint_data()
            self.timepoint_keys = [k for k in sorted(timepoint_data.keys()) 
                                   if not any(x in k for x in ['_', 'time_labels'])]
            self.num_timepoints = len(self.timepoint_keys)
            print(f"Training sequential growth for {self.num_timepoints} timepoints: {self.timepoint_keys}")

    def _compute_loss(self, main_batch, metric_samples_batch=None, validation=False):
        """Compute loss for sequential growth between timepoints."""
        x0s = main_batch["x0"][0]
        w0s = main_batch["x0"][1]
        
        # Setup metric sample pairs
        if self.args.manifold:
            if self.metric_clusters == 2:
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1])
                ] * self.branches
            else:
                branch_sample_pairs = []
                for b in range(self.branches):
                    if b + 1 < len(metric_samples_batch):
                        branch_sample_pairs.append(
                            (metric_samples_batch[0], metric_samples_batch[b + 1])
                        )
                    else:
                        branch_sample_pairs.append(
                            (metric_samples_batch[0], metric_samples_batch[1])
                        )
        
        total_loss = 0
        total_energy_loss = 0
        total_mass_loss = 0
        total_match_loss = 0
        total_recons_loss = 0
        num_transitions = 0
        
        # Process each consecutive timepoint transition
        for i in range(len(self.timepoint_keys) - 1):
            t_curr_key = self.timepoint_keys[i]
            t_next_key = self.timepoint_keys[i + 1]
            
            batch_curr_key = f"x{t_curr_key.replace('t', '').replace('final', '1')}"
            x_curr = main_batch[batch_curr_key][0]
            w_curr = main_batch[batch_curr_key][1]
            
            if i == len(self.timepoint_keys) - 2:
                # Final transition to branches
                # Get cluster size weights if available
                if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'cluster_sizes'):
                    cluster_sizes = self.trainer.datamodule.cluster_sizes
                    max_size = max(cluster_sizes)
                    # Inverse weighting: smaller clusters get higher weight
                    branch_weights = [max_size / size for size in cluster_sizes]
                else:
                    branch_weights = [1.0] * self.branches
                
                for b in range(self.branches):
                    x_next = main_batch[f"x1_{b+1}"][0]
                    w_next = main_batch[f"x1_{b+1}"][1]
                    
                    # Compute growth-based loss for this transition
                    loss, energy_l, mass_l, match_l, recons_l = self._compute_transition_loss(
                        x_curr, w_curr, x_next, w_next, b, i, 
                        branch_sample_pairs[b] if self.args.manifold else None
                    )
                    # Apply branch weight
                    total_loss += loss * branch_weights[b]
                    total_energy_loss += energy_l * branch_weights[b]
                    total_mass_loss += mass_l * branch_weights[b]
                    total_match_loss += match_l * branch_weights[b]
                    total_recons_loss += recons_l * branch_weights[b]
                    num_transitions += 1
            else:
                # Regular consecutive timepoints
                batch_next_key = f"x{t_next_key.replace('t', '').replace('final', '1')}"
                x_next = main_batch[batch_next_key][0]
                w_next = main_batch[batch_next_key][1]
                
                for b in range(self.branches):
                    loss, energy_l, mass_l, match_l, recons_l = self._compute_transition_loss(
                        x_curr, w_curr, x_next, w_next, b, i,
                        branch_sample_pairs[b] if self.args.manifold else None
                    )
                    total_loss += loss
                    total_energy_loss += energy_l
                    total_mass_loss += mass_l
                    total_match_loss += match_l
                    total_recons_loss += recons_l
                    num_transitions += 1
        
        # Average losses
        avg_energy_loss = total_energy_loss / num_transitions if num_transitions > 0 else total_energy_loss
        avg_mass_loss = total_mass_loss / num_transitions if num_transitions > 0 else total_mass_loss
        avg_match_loss = total_match_loss / num_transitions if num_transitions > 0 else total_match_loss
        avg_recons_loss = total_recons_loss / num_transitions if num_transitions > 0 else total_recons_loss
        
        # Log individual components
        if self.joint:
            if validation:
                self.log("JointTrain/val_energy_loss", avg_energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/val_mass_loss", avg_mass_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/val_match_loss", avg_match_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/val_recons_loss", avg_recons_loss, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log("JointTrain/train_energy_loss", avg_energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/train_mass_loss", avg_mass_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/train_match_loss", avg_match_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("JointTrain/train_recons_loss", avg_recons_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            if validation:
                self.log("GrowthNet/val_energy_loss", avg_energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/val_mass_loss", avg_mass_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/val_match_loss", avg_match_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/val_recons_loss", avg_recons_loss, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log("GrowthNet/train_energy_loss", avg_energy_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/train_mass_loss", avg_mass_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/train_match_loss", avg_match_loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log("GrowthNet/train_recons_loss", avg_recons_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss

    def _compute_transition_loss(self, x0, w0, x1, w1, branch_idx, transition_idx, metric_pair):
        """Compute loss for a single timepoint transition."""
        if self.ot_sampler is not None:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)
        
        # Simulate trajectory using flow network
        t_span = torch.linspace(0, 1, 10, device=x0.device)
        
        flow_model = flow_model_torch_wrapper(self.flow_nets[branch_idx])
        node = NeuralODE(flow_model, solver="euler", sensitivity="adjoint")
        
        with torch.no_grad():
            traj = node.trajectory(x0, t_span)
        
        # Compute energy and mass losses
        energy_loss = 0
        mass_loss = 0
        neg_weight_penalty = 0
        
        for t_idx in range(len(t_span)):
            t = t_span[t_idx]
            xt = traj[t_idx]
            
            # Growth rate
            growth = self.growth_nets[branch_idx](t.unsqueeze(0).expand(xt.shape[0]), xt)
            
            # Energy loss
            if self.args.manifold and metric_pair is not None:
                start_samples, end_samples = metric_pair
                samples = torch.cat([start_samples, end_samples], dim=0)
                _, kinetic, potential = self.data_manifold_metric.calculate_velocity(
                    xt, torch.zeros_like(xt), samples, transition_idx
                )
                energy = kinetic + potential
            else:
                energy = (growth ** 2).sum(dim=-1)
            
            energy_loss += energy.mean()
            
            # Mass conservation
            growth_sum = growth.sum(dim=-1, keepdim=True)  # Keep dimension for proper broadcasting
            wt = w0 * torch.exp(growth_sum)
            mass = wt.sum()
            mass_loss += (mass - w1.sum()).abs()
            neg_weight_penalty += torch.relu(-wt).sum()
        
        # Match and reconstruction losses (computed at final time)
        xt_final = traj[-1]
        match_loss = mean_squared_error(wt, w1)
        recons_loss = self.recons_loss(xt_final, x1)
        
        total_loss = (
            self.lambda_energy * energy_loss + 
            self.lambda_mass * (mass_loss + neg_weight_penalty) +
            self.lambda_match * match_loss +
            self.lambda_recons * recons_loss
        )
        
        return total_loss, energy_loss, mass_loss + neg_weight_penalty, match_loss, recons_loss

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        main_batch = batch["train_samples"]
        metric_batch = batch["metric_samples"]
        if isinstance(main_batch, tuple):
            main_batch = main_batch[0]
        if isinstance(metric_batch, tuple):
            metric_batch = metric_batch[0]
        
        loss = self._compute_loss(main_batch, metric_batch)
        
        if self.joint:
            self.log(
                "JointTrain/train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                "GrowthNet/train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        main_batch = batch["val_samples"]
        metric_batch = batch["metric_samples"]
        if isinstance(main_batch, tuple):
            main_batch = main_batch[0]
        if isinstance(metric_batch, tuple):
            metric_batch = metric_batch[0]
        
        loss = self._compute_loss(main_batch, metric_batch, validation=True)
        
        if self.joint:
            self.log(
                "JointTrain/val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        else:
            self.log(
                "GrowthNet/val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        
        return loss

    def configure_optimizers(self):
        import itertools
        params = list(itertools.chain(*[net.parameters() for net in self.growth_nets]))
        if self.joint:
            params += list(itertools.chain(*[net.parameters() for net in self.flow_nets]))
        
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                params, 
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        return optimizer