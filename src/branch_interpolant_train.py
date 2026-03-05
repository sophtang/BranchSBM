import sys
import os
import torch
import pytorch_lightning as pl
from .ema import EMA
import itertools
from .utils import plot_lidar
import matplotlib.pyplot as plt

class BranchInterpolantTrain(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        args,
        skipped_time_points: list = None,
        ot_sampler=None,
        
        state_cost=None,
        data_manifold_metric=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        self.flow_matcher = flow_matcher
        
        # list of geopath nets
        self.geopath_nets = flow_matcher.geopath_nets
        self.branches = len(self.geopath_nets)
        self.metric_clusters = args.metric_clusters
        
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points if skipped_time_points else []
        self.optimizer_name = args.geopath_optimizer
        self.lr = args.geopath_lr
        self.weight_decay = args.geopath_weight_decay
        self.args = args
        self.multiply_validation = 4

        self.first_loss = None
        self.timesteps = None
        self.computing_reference_loss = False
        
        # updates
        self.state_cost = state_cost
        self.data_manifold_metric = data_manifold_metric
        self.whiten = args.whiten

    def forward(self, x0, x1, t, branch_idx):
        # return specific branch interpolant
        return self.geopath_nets[branch_idx](x0, x1, t)

    def on_train_start(self):
        self.first_loss = self.compute_initial_loss()
        print("first loss")
        print(self.first_loss)

    # to edit
    def compute_initial_loss(self):
        # Set all GeoPath networks to eval mode
        for net in self.geopath_nets:
            net.train(mode=False)
        
        total_loss = 0
        total_count = 0
        with torch.enable_grad():
            self.t_val = []
            for i in range(
                self.trainer.datamodule.num_timesteps - len(self.skipped_time_points)
            ):
                self.t_val.append(
                    torch.rand(
                        self.trainer.datamodule.batch_size * self.multiply_validation,
                        requires_grad=True,
                    )
                )
        self.computing_reference_loss = True
        with torch.no_grad():
            old_alpha = self.flow_matcher.alpha
            self.flow_matcher.alpha = 0
            for batch in self.trainer.datamodule.train_dataloader():
                
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                if isinstance(batch, dict) and "train_samples" in batch:
                    main_batch_init = batch["train_samples"]
                    metric_batch_init = batch["metric_samples"]
                    if isinstance(main_batch_init, tuple):
                        main_batch_init = main_batch_init[0]
                    if isinstance(metric_batch_init, tuple):
                        metric_batch_init = metric_batch_init[0]
                else:
                    main_batch_init = batch
                    metric_batch_init = []
                
                self.timesteps = torch.linspace(
                    0.0, 1.0, len(main_batch_init["x0"])
                ).tolist()
                
                loss = self._compute_loss(
                    main_batch_init,
                    metric_batch_init,
                )
                print("initial loss")
                print(loss)
                # Skip NaN/Inf batches to prevent poisoning the average
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    total_count += 1
            self.flow_matcher.alpha = old_alpha
            
        self.computing_reference_loss = False
        
        # Set all GeoPath networks back to training mode
        for net in self.geopath_nets:
            net.train(mode=True)
        return total_loss / total_count if total_count > 0 else 1.0

    def _compute_loss(self, main_batch, metric_samples_batch=None):
        
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
        
        if self.args.manifold:
            #changed
            if self.metric_clusters == 7:
                # For 6 branches with 7 clusters (1 root + 6 branch endpoints)
                branch_sample_pairs = [
                    (metric_samples_batch[0], metric_samples_batch[1]),  # x0 → x1_1
                    (metric_samples_batch[0], metric_samples_batch[2]),  # x0 → x1_2
                    (metric_samples_batch[0], metric_samples_batch[3]),  # x0 → x1_3
                    (metric_samples_batch[0], metric_samples_batch[4]),  # x0 → x1_4
                    (metric_samples_batch[0], metric_samples_batch[5]),  # x0 → x1_5
                    (metric_samples_batch[0], metric_samples_batch[6]),  # x0 → x1_6
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
                    (metric_samples_batch[0], metric_samples_batch[0]),  # x0 → x1_1 (branch 1)
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
            """samples0, samples1, samples2 = (
                metric_samples_batch[0],
                metric_samples_batch[1],
                metric_samples_batch[2]
            )"""
                            
        assert len(x1s_list) == self.branches, "Mismatch between x1s_list and expected branches"
        
        # compute sum of velocities for each branch
        loss = 0
        velocities = []
        for branch_idx in range(self.branches):
        
            ts, xts, uts = self._process_flow(x0s, x1s_list[branch_idx], branch_idx)
            
            for i in range(len(ts)):
                # calculate kinetic and potential energy of the predicted interpolant
                if self.args.manifold:
                    start_samples, end_samples = branch_sample_pairs[branch_idx]
                    
                    samples = torch.cat([start_samples, end_samples], dim=0)
                    #print("metric sample shape")
                    #print(samples.shape)
                    vel, _, _ = self.data_manifold_metric.calculate_velocity(
                        xts[i], uts[i], samples, i
                    )
                else:
                    vel = torch.sqrt((uts[i]**2).sum(dim =-1) + self.state_cost(xts[i]))
                    #vel = (uts[i]**2).sum(dim =-1)
                
                velocities.append(vel)
            
        velocity_loss = torch.mean(torch.cat(velocities) ** 2)
        
        self.log(
            "BranchPathNet/mean_velocity_geopath",
            velocity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        return velocity_loss

    def _process_flow(self, x0s, x1s, branch_idx):
        ts, xts, uts = [], [], []
        t_start = self.timesteps[0]
        i_start = 0

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)
            if self.trainer.validating or self.computing_reference_loss:
                repeat_tuple = (self.multiply_validation, 1) + (1,) * (
                    len(x0.shape) - 2
                )
                x0 = x0.repeat(repeat_tuple)
                x1 = x1.repeat(repeat_tuple)

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

            t = None
            if self.trainer.validating or self.computing_reference_loss:
                t = self.t_val[i]

            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t_start, t_start_next, branch_idx, training_geopath_net=True, t=t
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
            metric_batch = batch["metric_samples"]
            if isinstance(main_batch, tuple):
                main_batch = main_batch[0]
            if isinstance(metric_batch, tuple):
                metric_batch = metric_batch[0]
        else:
            # Fallback
            main_batch = batch.get("train_samples", batch)
            metric_batch = batch.get("metric_samples", [])
        
        # Debug: print structure
        if batch_idx == 0:
            print(f"DEBUG batch type: {type(batch)}")
            if isinstance(batch, dict):
                print(f"DEBUG batch keys: {batch.keys()}")
                print(f"DEBUG train_samples type: {type(batch.get('train_samples'))}")
                if isinstance(batch.get("train_samples"), dict):
                    print(f"DEBUG train_samples keys: {batch['train_samples'].keys()}")
                    print(f"DEBUG x0 type: {type(batch['train_samples'].get('x0'))}")
                    if 'x0' in batch['train_samples']:
                        x0_item = batch['train_samples']['x0']
                        print(f"DEBUG x0 structure: {type(x0_item)}")
                        if isinstance(x0_item, (list, tuple)):
                            print(f"DEBUG x0 length: {len(x0_item)}")
                            if len(x0_item) > 0:
                                print(f"DEBUG x0[0] shape: {x0_item[0].shape if hasattr(x0_item[0], 'shape') else 'no shape'}")
            print(f"DEBUG main_batch type: {type(main_batch)}")
            if isinstance(main_batch, dict):
                print(f"DEBUG main_batch keys: {main_batch.keys()}")
        
        self.timesteps = torch.linspace(0.0, 1.0, len(main_batch["x0"])).tolist()
        tangential_velocity_loss = self._compute_loss(main_batch, metric_batch)
        
        if self.first_loss:
            tangential_velocity_loss = tangential_velocity_loss / self.first_loss
            
        self.log(
            "BranchPathNet/mean_geopath_geopath",
            (self.flow_matcher.geopath_net_output.abs().mean()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        
        self.log(
            "BranchPathNet/train_loss_geopath",
            tangential_velocity_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
                
        return tangential_velocity_loss

    def validation_step(self, batch, batch_idx):
        # Handle both dict and tuple batch formats from CombinedLoader
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
        tangential_velocity_loss = self._compute_loss(main_batch, metric_batch)
        if self.first_loss:
            tangential_velocity_loss = tangential_velocity_loss / self.first_loss
            
        self.log(
            "BranchPathNet/val_loss_geopath",
            tangential_velocity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return tangential_velocity_loss
    
    
    def test_step(self, batch, batch_idx):
        # Handle both tuple and dict batch formats from CombinedLoader
        if isinstance(batch, dict):
            main_batch = batch["test_samples"]
            metric_batch = batch["metric_samples"]
            # CombinedLoader may wrap values in a tuple
            if isinstance(main_batch, tuple):
                main_batch = main_batch[0]
            if isinstance(metric_batch, tuple):
                metric_batch = metric_batch[0]
        else:
            # batch is a tuple: (test_samples, metric_samples)
            main_batch = batch[0][0]
            metric_batch = batch[1][0]
                
        x0 = main_batch["x0"][0]  # [B, D]
        cloud_points = main_batch["dataset"][0]  # full dataset, [N, D]
        
        x0 = x0.to(self.device)
        cloud_points = cloud_points.to(self.device)

        t_vals = [0.25, 0.5, 0.75]
        t_labels = ["t=1/4", "t=1/2", "t=3/4"]

        colors = {
            "x0": "#4D176C",
            "t=1/4": "#5C3B9D",
            "t=1/2": "#6172B9",
            "t=3/4": "#AC4E51",
            "x1": "#771F4F",
        }

        # Unwhiten cloud points if needed
        if self.whiten:
            cloud_points = torch.tensor(
                self.trainer.datamodule.scaler.inverse_transform(cloud_points.cpu().numpy())
            )

        for i in range(self.branches):
            geopath = self.geopath_nets[i]
            x1_key = f"x1_{i + 1}"
            if x1_key not in main_batch:
                print(f"Skipping branch {i + 1}: no final distribution {x1_key}")
                continue

            x1 = main_batch[x1_key][0].to(self.device)
            print(x1.shape)
            print(x0.shape)
            interpolated_points = []
            with torch.no_grad():
                for t_scalar in t_vals:
                    t_tensor = torch.full((x0.shape[0], 1), t_scalar, device=self.device)  # [B, 1]
                    xt = geopath(x0, x1, t_tensor).cpu()  # [B, D]
                    if self.whiten:
                        xt = torch.tensor(
                            self.trainer.datamodule.scaler.inverse_transform(xt.numpy())
                        )
                    interpolated_points.append(xt)

            if self.whiten:
                x0_plot = torch.tensor(
                    self.trainer.datamodule.scaler.inverse_transform(x0.cpu().numpy())
                )
                x1_plot = torch.tensor(
                    self.trainer.datamodule.scaler.inverse_transform(x1.cpu().numpy())
                )
            else:
                x0_plot = x0.cpu()
                x1_plot = x1.cpu()

            # Plot
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
            ax.view_init(elev=30, azim=-115, roll=0)
            plot_lidar(ax, cloud_points)

            # Initial x₀
            ax.scatter(
                x0_plot[:, 0], x0_plot[:, 1], x0_plot[:, 2],
                s=15, alpha=1.0, color=colors["x0"], label="x₀", depthshade=True,
                edgecolors="white",
                linewidths=0.3
            )

            # Interpolated points
            for xt, t_label in zip(interpolated_points, t_labels):
                ax.scatter(
                    xt[:, 0], xt[:, 1], xt[:, 2],
                    s=15, alpha=1.0, color=colors[t_label], label=t_label, depthshade=True,
                    edgecolors="white",
                    linewidths=0.3
                )

            # Final x₁
            ax.scatter(
                x1_plot[:, 0], x1_plot[:, 1], x1_plot[:, 2],
                s=15, alpha=1.0, color=colors["x1"], label="x₁", depthshade=True,
                edgecolors="white",
                linewidths=0.3
            )

            ax.legend()
            
            # Use consistent path structure for results
            run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
            results_dir = os.path.join(self.args.working_dir, 'results', run_name)
            figures_dir = os.path.join(results_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            
            save_path = f"{figures_dir}/lidar_geopath_branch_{i+1}.png"
            plt.savefig(save_path, dpi=300)
            plt.close()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        for net in self.geopath_nets:
            if isinstance(net, EMA):
                net.update_ema()

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                itertools.chain(*[net.parameters() for net in self.geopath_nets]), lr=self.lr
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                itertools.chain(*[net.parameters() for net in self.geopath_nets]), lr=self.lr
            )
        return optimizer
