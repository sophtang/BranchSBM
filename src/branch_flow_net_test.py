"""
Separate test classes for each BranchSBM experiment with specific plotting styles.
Each class handles testing and visualization for: LiDAR, Mouse, Clonidine, Trametinib, Veres.
"""

import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import random
import ot
from torchdyn.core import NeuralODE
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from .networks.utils import flow_model_torch_wrapper
from .branch_flow_net_train import BranchFlowNetTrainBase
from .branch_growth_net_train import GrowthNetTrain
from .utils import wasserstein, mix_rbf_mmd2, plot_lidar
import json

def evaluate_model(gt_data, model_data, a, b):
    # ensure inputs are tensors
    if not isinstance(gt_data, torch.Tensor):
        gt_data = torch.tensor(gt_data, dtype=torch.float32)
    if not isinstance(model_data, torch.Tensor):
        model_data = torch.tensor(model_data, dtype=torch.float32)

    # choose device: prefer model_data's device if it's not CPU, otherwise use gt_data's device
    try:
        model_dev = model_data.device
    except Exception:
        model_dev = torch.device('cpu')
    try:
        gt_dev = gt_data.device
    except Exception:
        gt_dev = torch.device('cpu')

    device = model_dev if model_dev.type != 'cpu' else gt_dev

    gt = gt_data.to(device=device, dtype=torch.float32)
    md = model_data.to(device=device, dtype=torch.float32)

    M = torch.cdist(gt, md, p=2).cpu().numpy()
    if np.isnan(M).any() or np.isinf(M).any():
        return np.nan
    return ot.emd2(a, b, M, numItermax=1e7)

def compute_distribution_distances(pred, true, pred_full=None, true_full=None):
    w1 = wasserstein(pred, true, power=1)
    w2 = wasserstein(pred, true, power=2)
    
    # Use full dimensions for MMD if provided, otherwise use same as W1/W2
    mmd_pred = pred_full if pred_full is not None else pred
    mmd_true = true_full if true_full is not None else true
    
    # MMD requires same number of samples — randomly subsample the larger set
    n_pred, n_true = mmd_pred.shape[0], mmd_true.shape[0]
    if n_pred > n_true:
        perm = torch.randperm(n_pred)[:n_true]
        mmd_pred = mmd_pred[perm]
    elif n_true > n_pred:
        perm = torch.randperm(n_true)[:n_pred]
        mmd_true = mmd_true[perm]
    mmd = mix_rbf_mmd2(mmd_pred, mmd_true, sigma_list=[0.01, 0.1, 1, 10, 100]).item()
    
    return {"W1": w1, "W2": w2, "MMD": mmd}


def compute_tmv_from_mass_over_time(mass_over_time, all_endpoints, time_points=None, timepoint_data=None, time_index=None, target_time=None, gt_key_template='t1_{}', weights_over_time=None):

    if weights_over_time is not None or mass_over_time is not None:
        if time_index is None:
            if target_time is not None and time_points is not None:
                arr = np.array(time_points)
                time_index = int(np.argmin(np.abs(arr - float(target_time))))
            else:
                # default to last index
                ref_list = weights_over_time if weights_over_time is not None else mass_over_time
                time_index = len(ref_list[0]) - 1
    else:
        # neither available; time_index not used
        if time_index is None:
            time_index = -1

    n_branches = len(all_endpoints)

    # initial total cells for normalization
    n_initial = None
    if timepoint_data is not None and 't0' in timepoint_data:
        try:
            n_initial = int(timepoint_data['t0'].shape[0])
        except Exception:
            n_initial = None

    pred_masses = []
    for i in range(n_branches):
        # Use sum of actual particle weights if available, otherwise mean_weight * num_particles
        if weights_over_time is not None:
            try:
                weights_tensor = weights_over_time[i][time_index]
                # Sum all particle weights to get total mass for this branch
                total_mass = float(weights_tensor.sum().item())
                pred_masses.append(total_mass)
                continue
            except Exception:
                pass  # Fall through to mean weight calculation
        
        # Fallback: mean weight from mass_over_time if available, otherwise assume weight=1
        mean_w = 1.0
        if mass_over_time is not None:
            try:
                mean_w = float(mass_over_time[i][time_index])
            except Exception:
                mean_w = 1.0

        # determine number of particles for this branch
        num_particles = 0
        try:
            if hasattr(all_endpoints[i], 'shape'):
                num_particles = int(all_endpoints[i].shape[0])
            else:
                num_particles = int(len(all_endpoints[i]))
        except Exception:
            num_particles = 0

        pred_masses.append(mean_w * float(num_particles))

    # ground-truth masses per branch
    gt_masses = []
    if timepoint_data is not None:
        for i in range(n_branches):
            key1 = gt_key_template.format(i)
            if key1 in timepoint_data:
                gt_masses.append(float(timepoint_data[key1].shape[0]))
            else:
                base_key = gt_key_template.split("_")[0] if '_' in gt_key_template else gt_key_template
                if base_key in timepoint_data:
                    gt_masses.append(float(timepoint_data[base_key].shape[0]))
                else:
                    gt_masses.append(0.0)
    else:
        gt_masses = [0.0 for _ in range(n_branches)]

    # determine normalization denominator
    if n_initial is None:
        s = float(sum(gt_masses))
        if s > 0:
            n_initial = s
        else:
            n_initial = float(sum(pred_masses)) if sum(pred_masses) > 0 else 1.0

    pred_fracs = [m / float(n_initial) for m in pred_masses]
    gt_fracs = [m / float(n_initial) for m in gt_masses]

    tmv = 0.5 * float(np.sum(np.abs(np.array(pred_fracs) - np.array(gt_fracs))))

    return {
        'time_index': time_index,
        'pred_masses': pred_masses,
        'gt_masses': gt_masses,
        'pred_fracs': pred_fracs,
        'gt_fracs': gt_fracs,
        'tmv': tmv,
    }


class FlowNetTestLidar(GrowthNetTrain):
    
    def test_step(self, batch, batch_idx):
        # Unwrap CombinedLoader outer tuple if needed
        if isinstance(batch, (list, tuple)) and len(batch) == 1:
            batch = batch[0]
        
        if isinstance(batch, dict) and "test_samples" in batch:
            test_samples = batch["test_samples"]
            metric_samples = batch["metric_samples"]
            
            if isinstance(test_samples, (list, tuple)) and len(test_samples) >= 2 and isinstance(test_samples[-1], int):
                test_samples = test_samples[0]
            if isinstance(metric_samples, (list, tuple)) and len(metric_samples) >= 2 and isinstance(metric_samples[-1], int):
                metric_samples = metric_samples[0]
            
            if isinstance(test_samples, (list, tuple)) and len(test_samples) == 1:
                test_samples = test_samples[0]
            main_batch = test_samples
                
            if isinstance(metric_samples, dict):
                metric_batch = list(metric_samples.values())
            elif isinstance(metric_samples, (list, tuple)):
                metric_batch = [m[0] if isinstance(m, (list, tuple)) and len(m) == 1 else m for m in metric_samples]
            else:
                metric_batch = [metric_samples]
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            # Old tuple format: (test_samples, metric_samples)
            # Each could be dict or list
            test_samples = batch[0]
            metric_samples = batch[1]
            
            if isinstance(test_samples, dict):
                main_batch = test_samples
            elif isinstance(test_samples, (list, tuple)):
                main_batch = test_samples[0]
            else:
                main_batch = test_samples
            
            if isinstance(metric_samples, dict):
                metric_batch = list(metric_samples.values())
            elif isinstance(metric_samples, (list, tuple)):
                metric_batch = [m[0] if isinstance(m, (list, tuple)) and len(m) == 1 else m for m in metric_samples]
            else:
                metric_batch = [metric_samples]
        else:
            # Fallback
            main_batch = batch
            metric_batch = []
        
        timepoint_data = self.trainer.datamodule.get_timepoint_data()
        # main_batch is a dict like {"x0": (tensor, weights), ...}
        if isinstance(main_batch, dict):
            device = main_batch["x0"][0].device
        else:
            device = main_batch[0]["x0"][0].device
        
        x0_all = self.trainer.datamodule.val_dataloaders["x0"].dataset.tensors[0].to(device)
        w0_all = torch.ones(x0_all.shape[0], 1, dtype=torch.float32).to(device)
        full_batch = {"x0": (x0_all, w0_all)}
        
        time_points, all_endpoints, all_trajs, mass_over_time, energy_over_time, weights_over_time = self.get_mass_and_position(full_batch, metric_batch)
        
        cloud_points = main_batch["dataset"][0]  # [N, 3]

        # Run 5 trials with random subsampling for robust metrics
        n_trials = 5

        # Compute per-branch metrics
        metrics_dict = {}
        for i, endpoints in enumerate(all_endpoints):
            true_data_key = f't1_{i+1}' if f't1_{i+1}' in timepoint_data else 't1'
            true_data = torch.tensor(timepoint_data[true_data_key], dtype=torch.float32).to(endpoints.device)
            
            w1_br, w2_br, mmd_br = [], [], []
            for trial in range(n_trials):
                n_min = min(endpoints.shape[0], true_data.shape[0])
                perm_pred = torch.randperm(endpoints.shape[0])[:n_min]
                perm_gt = torch.randperm(true_data.shape[0])[:n_min]
                m = compute_distribution_distances(
                    endpoints[perm_pred, :2], true_data[perm_gt, :2],
                    pred_full=endpoints[perm_pred], true_full=true_data[perm_gt]
                )
                w1_br.append(m["W1"]); w2_br.append(m["W2"]); mmd_br.append(m["MMD"])
            
            metrics_dict[f"branch_{i+1}"] = {
                "W1_mean": float(np.mean(w1_br)), "W1_std": float(np.std(w1_br, ddof=1)),
                "W2_mean": float(np.mean(w2_br)), "W2_std": float(np.std(w2_br, ddof=1)),
                "MMD_mean": float(np.mean(mmd_br)), "MMD_std": float(np.std(mmd_br, ddof=1)),
            }
            self.log(f"test/W1_branch{i+1}", np.mean(w1_br), on_epoch=True)
            print(f"Branch {i+1} — W1: {np.mean(w1_br):.6f}±{np.std(w1_br, ddof=1):.6f}, "
                  f"W2: {np.mean(w2_br):.6f}±{np.std(w2_br, ddof=1):.6f}, "
                  f"MMD: {np.mean(mmd_br):.6f}±{np.std(mmd_br, ddof=1):.6f}")
        
        # Compute combined metrics across all branches (5 trials)
        all_pred_combined = torch.cat(list(all_endpoints), dim=0)
        all_true_list = []
        for i in range(len(all_endpoints)):
            true_data_key = f't1_{i+1}' if f't1_{i+1}' in timepoint_data else 't1'
            all_true_list.append(torch.tensor(timepoint_data[true_data_key], dtype=torch.float32).to(all_pred_combined.device))
        all_true_combined = torch.cat(all_true_list, dim=0)
        
        w1_trials, w2_trials, mmd_trials = [], [], []
        for trial in range(n_trials):
            n_min = min(all_pred_combined.shape[0], all_true_combined.shape[0])
            perm_pred = torch.randperm(all_pred_combined.shape[0])[:n_min]
            perm_gt = torch.randperm(all_true_combined.shape[0])[:n_min]
            m = compute_distribution_distances(
                all_pred_combined[perm_pred, :2], all_true_combined[perm_gt, :2],
                pred_full=all_pred_combined[perm_pred], true_full=all_true_combined[perm_gt]
            )
            w1_trials.append(m["W1"]); w2_trials.append(m["W2"]); mmd_trials.append(m["MMD"])

        w1_mean, w1_std = np.mean(w1_trials), np.std(w1_trials, ddof=1)
        w2_mean, w2_std = np.mean(w2_trials), np.std(w2_trials, ddof=1)
        mmd_mean, mmd_std = np.mean(mmd_trials), np.std(mmd_trials, ddof=1)
        self.log("test/W1_combined", w1_mean, on_epoch=True)
        self.log("test/W2_combined", w2_mean, on_epoch=True)
        self.log("test/MMD_combined", mmd_mean, on_epoch=True)
        
        metrics_dict["combined"] = {
            "W1_mean": float(w1_mean), "W1_std": float(w1_std),
            "W2_mean": float(w2_mean), "W2_std": float(w2_std),
            "MMD_mean": float(mmd_mean), "MMD_std": float(mmd_std),
            "n_trials": n_trials,
        }
        print(f"\n=== Combined ===")
        print(f"W1: {w1_mean:.6f} ± {w1_std:.6f}")
        print(f"W2: {w2_mean:.6f} ± {w2_std:.6f}")
        print(f"MMD: {mmd_mean:.6f} ± {mmd_std:.6f}")

        # Inverse-transform cloud points for visualization
        if self.whiten:
            cloud_points = torch.tensor(
                self.trainer.datamodule.scaler.inverse_transform(
                    cloud_points.cpu().detach().numpy()
                )
            )

        # Create results directory structure
        run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
        results_dir = os.path.join(self.args.working_dir, 'results', run_name)
        figures_dir = f'{results_dir}/figures'
        os.makedirs(figures_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = f'{results_dir}/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        
        # Save detailed per-branch metrics to CSV
        detailed_csv_path = f'{results_dir}/metrics_detailed.csv'
        with open(detailed_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric_Group', 'W1_Mean', 'W1_Std', 'W2_Mean', 'W2_Std', 'MMD_Mean', 'MMD_Std'])
            for key in sorted(metrics_dict.keys()):
                m = metrics_dict[key]
                writer.writerow([key,
                    f'{m.get("W1_mean", m.get("W1", 0)):.6f}', f'{m.get("W1_std", 0):.6f}',
                    f'{m.get("W2_mean", m.get("W2", 0)):.6f}', f'{m.get("W2_std", 0):.6f}',
                    f'{m.get("MMD_mean", m.get("MMD", 0)):.6f}', f'{m.get("MMD_std", 0):.6f}'])
        print(f"Detailed metrics CSV saved to {detailed_csv_path}")

        # Convert all_trajs from list of lists to stacked tensors for plotting
        # all_trajs[i] is a list of T tensors of shape [B, D]
        # Stack to get shape [B, T, D]
        stacked_trajs = []
        for traj_list in all_trajs:
            # Stack along time dimension (dim=1) to get [B, T, D]
            stacked_traj = torch.stack(traj_list, dim=1)
            stacked_trajs.append(stacked_traj)
        
        # Inverse-transform trajectories to match cloud_points coordinates
        if self.whiten:
            stacked_trajs_original = []
            for traj in stacked_trajs:
                B, T, D = traj.shape
                # Reshape to [B*T, D] for inverse transform
                traj_flat = traj.reshape(-1, D).cpu().detach().numpy()
                traj_inv = self.trainer.datamodule.scaler.inverse_transform(traj_flat)
                # Reshape back to [B, T, D]
                traj_inv = torch.tensor(traj_inv).reshape(B, T, D)
                stacked_trajs_original.append(traj_inv)
            stacked_trajs = stacked_trajs_original

        # ===== Plot all branches together =====
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.view_init(elev=30, azim=-115, roll=0)
        for i, traj in enumerate(stacked_trajs):
            plot_lidar(ax, cloud_points, xs=traj, branch_idx=i)
        plt.savefig(f'{figures_dir}/{self.args.data_name}_all_branches.png', dpi=300)
        plt.close()

        # ===== Plot each branch separately =====
        for i, traj in enumerate(stacked_trajs):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
            ax.view_init(elev=30, azim=-115, roll=0)
            plot_lidar(ax, cloud_points, xs=traj, branch_idx=i)
            plt.savefig(f'{figures_dir}/{self.args.data_name}_branch_{i + 1}.png', dpi=300)
            plt.close()

        print(f"LiDAR figures saved to {figures_dir}")


class FlowNetTestMouse(GrowthNetTrain):
    
    def test_step(self, batch, batch_idx):
        # Handle both tuple and dict batch formats from CombinedLoader
        if isinstance(batch, dict):
            main_batch = batch.get("test_samples", batch)
            if isinstance(main_batch, tuple):
                main_batch = main_batch[0]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
            if isinstance(batch[0], dict):
                main_batch = batch[0].get("test_samples", batch[0])
                if isinstance(main_batch, tuple):
                    main_batch = main_batch[0]
            else:
                main_batch = batch[0][0]
        else:
            main_batch = batch
        
        device = main_batch["x0"][0].device
        
        # Use val x0 as initial conditions
        x0 = self.trainer.datamodule.val_dataloaders["x0"].dataset.tensors[0].to(device)
        
        # Get timepoint data for ground truth
        timepoint_data = self.trainer.datamodule.get_timepoint_data()
        
        # Ground truth at t1 (intermediate timepoint)
        data_t1 = torch.tensor(timepoint_data['t1'], dtype=torch.float32)
        
        # Define color schemes for mouse (2 branches)
        custom_colors_1 = ["#05009E", "#A19EFF", "#B83CFF"]
        custom_colors_2 = ["#05009E", "#A19EFF", "#50B2D7"]
        custom_cmap_1 = LinearSegmentedColormap.from_list("cmap1", custom_colors_1)
        custom_cmap_2 = LinearSegmentedColormap.from_list("cmap2", custom_colors_2)
        
        t_span_full = torch.linspace(0, 1.0, 100).to(device)
        all_trajs = []
        
        for i, flow_net in enumerate(self.flow_nets):
            node = NeuralODE(
                flow_model_torch_wrapper(flow_net),
                solver="euler",
                sensitivity="adjoint",
            ).to(device)
            
            with torch.no_grad():
                traj = node.trajectory(x0, t_span_full).cpu()  # [T, B, D]
            
            traj = torch.transpose(traj, 0, 1)  # [B, T, D]
            all_trajs.append(traj)
        
        t_span_metric_t1 = torch.linspace(0, 0.5, 50).to(device)
        t_span_metric_t2 = torch.linspace(0, 1.0, 100).to(device)
        n_trials = 5

        # Gather t2 branch ground truth
        data_t2_branches = []
        for i in range(len(self.flow_nets)):
            key = f't2_{i+1}'
            if key in timepoint_data:
                data_t2_branches.append(torch.tensor(timepoint_data[key], dtype=torch.float32))
            elif i == 0 and 't2' in timepoint_data:
                data_t2_branches.append(torch.tensor(timepoint_data['t2'], dtype=torch.float32))
            else:
                data_t2_branches.append(None)

        # Combined t2 ground truth (all branches merged)
        data_t2_all_list = [d for d in data_t2_branches if d is not None]
        data_t2_combined = torch.cat(data_t2_all_list, dim=0) if data_t2_all_list else None

        # ---- t1 combined metrics (all branches pooled, compared to t1) ----
        w1_t1_trials, w2_t1_trials, mmd_t1_trials = [], [], []

        for trial in range(n_trials):
            all_preds = []
            for i, flow_net in enumerate(self.flow_nets):
                node = NeuralODE(
                    flow_model_torch_wrapper(flow_net),
                    solver="euler",
                    sensitivity="adjoint",
                ).to(device)
                
                with torch.no_grad():
                    traj = node.trajectory(x0, t_span_metric_t1)  # [T, B, D]
                
                x_final = traj[-1].cpu()  # [B, D]
                all_preds.append(x_final)
            
            preds = torch.cat(all_preds, dim=0)
            target_size = preds.shape[0]
            perm = torch.randperm(data_t1.shape[0])[:target_size]
            data_t1_reduced = data_t1[perm]
            
            metrics = compute_distribution_distances(
                preds[:, :2], data_t1_reduced[:, :2]
            )
            w1_t1_trials.append(metrics["W1"])
            w2_t1_trials.append(metrics["W2"])
            mmd_t1_trials.append(metrics["MMD"])

        # ---- t2 per-branch metrics (each branch endpoint vs its own t2 cluster) ----
        branch_t2_metrics = {}
        for i, flow_net in enumerate(self.flow_nets):
            if data_t2_branches[i] is None:
                continue
            w1_br, w2_br, mmd_br = [], [], []
            for trial in range(n_trials):
                node = NeuralODE(
                    flow_model_torch_wrapper(flow_net),
                    solver="euler",
                    sensitivity="adjoint",
                ).to(device)
                with torch.no_grad():
                    traj = node.trajectory(x0, t_span_metric_t2)
                x_final = traj[-1].cpu()
                gt = data_t2_branches[i]
                n_min = min(x_final.shape[0], gt.shape[0])
                perm_pred = torch.randperm(x_final.shape[0])[:n_min]
                perm_gt = torch.randperm(gt.shape[0])[:n_min]
                m = compute_distribution_distances(
                    x_final[perm_pred, :2], gt[perm_gt, :2]
                )
                w1_br.append(m["W1"])
                w2_br.append(m["W2"])
                mmd_br.append(m["MMD"])
            branch_t2_metrics[f"branch_{i+1}_t2"] = {
                "W1_mean": float(np.mean(w1_br)), "W1_std": float(np.std(w1_br, ddof=1)),
                "W2_mean": float(np.mean(w2_br)), "W2_std": float(np.std(w2_br, ddof=1)),
                "MMD_mean": float(np.mean(mmd_br)), "MMD_std": float(np.std(mmd_br, ddof=1)),
            }
            print(f"Branch {i+1} @ t2 — W1: {np.mean(w1_br):.6f}±{np.std(w1_br, ddof=1):.6f}, "
                  f"W2: {np.mean(w2_br):.6f}±{np.std(w2_br, ddof=1):.6f}, "
                  f"MMD: {np.mean(mmd_br):.6f}±{np.std(mmd_br, ddof=1):.6f}")

        # ---- t2 combined metrics (all branches pooled, compared to all t2) ----
        w1_t2_trials, w2_t2_trials, mmd_t2_trials = [], [], []
        if data_t2_combined is not None:
            for trial in range(n_trials):
                all_preds = []
                for i, flow_net in enumerate(self.flow_nets):
                    node = NeuralODE(
                        flow_model_torch_wrapper(flow_net),
                        solver="euler",
                        sensitivity="adjoint",
                    ).to(device)
                    with torch.no_grad():
                        traj = node.trajectory(x0, t_span_metric_t2)
                    all_preds.append(traj[-1].cpu())
                preds = torch.cat(all_preds, dim=0)
                n_min = min(preds.shape[0], data_t2_combined.shape[0])
                perm_pred = torch.randperm(preds.shape[0])[:n_min]
                perm_gt = torch.randperm(data_t2_combined.shape[0])[:n_min]
                m = compute_distribution_distances(
                    preds[perm_pred, :2], data_t2_combined[perm_gt, :2]
                )
                w1_t2_trials.append(m["W1"])
                w2_t2_trials.append(m["W2"])
                mmd_t2_trials.append(m["MMD"])
        
        # Compute mean and std
        w1_t1_mean, w1_t1_std = np.mean(w1_t1_trials), np.std(w1_t1_trials, ddof=1)
        w2_t1_mean, w2_t1_std = np.mean(w2_t1_trials), np.std(w2_t1_trials, ddof=1)
        mmd_t1_mean, mmd_t1_std = np.mean(mmd_t1_trials), np.std(mmd_t1_trials, ddof=1)
        
        # Log metrics
        self.log("test/W1_combined_t1", w1_t1_mean, on_epoch=True)
        self.log("test/W2_combined_t1", w2_t1_mean, on_epoch=True)
        self.log("test/MMD_combined_t1", mmd_t1_mean, on_epoch=True)
        
        metrics_dict = {
            "combined_t1": {
                "W1_mean": float(w1_t1_mean), "W1_std": float(w1_t1_std),
                "W2_mean": float(w2_t1_mean), "W2_std": float(w2_t1_std),
                "MMD_mean": float(mmd_t1_mean), "MMD_std": float(mmd_t1_std),
                "n_trials": n_trials,
            }
        }
        metrics_dict.update(branch_t2_metrics)

        if w1_t2_trials:
            w1_t2_mean, w1_t2_std = np.mean(w1_t2_trials), np.std(w1_t2_trials, ddof=1)
            w2_t2_mean, w2_t2_std = np.mean(w2_t2_trials), np.std(w2_t2_trials, ddof=1)
            mmd_t2_mean, mmd_t2_std = np.mean(mmd_t2_trials), np.std(mmd_t2_trials, ddof=1)
            self.log("test/W1_combined_t2", w1_t2_mean, on_epoch=True)
            self.log("test/W2_combined_t2", w2_t2_mean, on_epoch=True)
            self.log("test/MMD_combined_t2", mmd_t2_mean, on_epoch=True)
            metrics_dict["combined_t2"] = {
                "W1_mean": float(w1_t2_mean), "W1_std": float(w1_t2_std),
                "W2_mean": float(w2_t2_mean), "W2_std": float(w2_t2_std),
                "MMD_mean": float(mmd_t2_mean), "MMD_std": float(mmd_t2_std),
                "n_trials": n_trials,
            }
        
        print(f"\n=== Combined @ t1 ===")
        print(f"W1: {w1_t1_mean:.6f} ± {w1_t1_std:.6f}")
        print(f"W2: {w2_t1_mean:.6f} ± {w2_t1_std:.6f}")
        print(f"MMD: {mmd_t1_mean:.6f} ± {mmd_t1_std:.6f}")
        if w1_t2_trials:
            print(f"\n=== Combined @ t2 ===")
            print(f"W1: {w1_t2_mean:.6f} ± {w1_t2_std:.6f}")
            print(f"W2: {w2_t2_mean:.6f} ± {w2_t2_std:.6f}")
            print(f"MMD: {mmd_t2_mean:.6f} ± {mmd_t2_std:.6f}")

        # Create results directory structure
        run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
        results_dir = os.path.join(self.args.working_dir, 'results', run_name)
        figures_dir = f'{results_dir}/figures'
        os.makedirs(figures_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = f'{results_dir}/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        
        
        # Save detailed metrics to CSV
        detailed_csv_path = f'{results_dir}/metrics_detailed.csv'
        with open(detailed_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric_Group', 'W1_Mean', 'W1_Std', 'W2_Mean', 'W2_Std', 'MMD_Mean', 'MMD_Std'])
            for key in sorted(metrics_dict.keys()):
                m = metrics_dict[key]
                writer.writerow([key,
                    f'{m.get("W1_mean", 0):.6f}', f'{m.get("W1_std", 0):.6f}',
                    f'{m.get("W2_mean", 0):.6f}', f'{m.get("W2_std", 0):.6f}',
                    f'{m.get("MMD_mean", 0):.6f}', f'{m.get("MMD_std", 0):.6f}'])
        print(f"Detailed metrics CSV saved to {detailed_csv_path}")

        # ===== Plot individual branches (using full t_span trajectories) =====
        self._plot_mouse_branches(all_trajs, timepoint_data, figures_dir, custom_cmap_1, custom_cmap_2)

        # ===== Plot all branches together =====
        self._plot_mouse_combined(all_trajs, timepoint_data, figures_dir, custom_cmap_1, custom_cmap_2)

        print(f"Mouse figures saved to {figures_dir}")

    def _plot_mouse_branches(self, all_trajs, timepoint_data, save_dir, cmap1, cmap2):
        """Plot each branch separately with timepoint background."""
        n_branches = len(all_trajs)
        branch_names = [f'Branch {i+1}' for i in range(n_branches)]
        branch_colors = ['#B83CFF', '#50B2D7'][:n_branches]
        cmaps = [cmap1, cmap2][:n_branches]
        
        # Stack list-of-tensors into [B, T, D] numpy arrays
        all_trajs_np = []
        for traj in all_trajs:
            if isinstance(traj, list):
                traj = torch.stack(traj, dim=1)  # list of [B,D] -> [B,T,D]
            all_trajs_np.append(traj.cpu().detach().numpy())
        all_trajs = all_trajs_np
        
        # Move timepoint data to numpy
        for key in list(timepoint_data.keys()):
            if torch.is_tensor(timepoint_data[key]):
                timepoint_data[key] = timepoint_data[key].cpu().numpy()
        
        # Compute global axis limits
        all_coords = []
        for key in ['t0', 't1', 't2', 't2_1', 't2_2']:
            if key in timepoint_data:
                all_coords.append(timepoint_data[key][:, :2])
        for traj_np in all_trajs:
            all_coords.append(traj_np.reshape(-1, traj_np.shape[-1])[:, :2])
        
        all_coords = np.concatenate(all_coords, axis=0)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        # Add margin
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        for i, traj in enumerate(all_trajs):
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = cmaps[i]
            c_end = branch_colors[i]
            
            # Plot timepoint background
            t2_key = f't2_{i+1}' if f't2_{i+1}' in timepoint_data else 't2'
            coords_list = [timepoint_data['t0'], timepoint_data['t1'], timepoint_data[t2_key]]
            tp_colors = ['#05009E', '#A19EFF', c_end]
            tp_labels = ["t=0", "t=1", f"t=2 (branch {i+1})"]
            
            for coords, color, label in zip(coords_list, tp_colors, tp_labels):
                alpha = 0.8 if color == '#05009E' else 0.6
                ax.scatter(coords[:, 0], coords[:, 1], 
                           c=color, s=80, alpha=alpha, marker='x', 
                           label=f'{label} cells', linewidth=1.5)
            
            # Plot continuous trajectories with LineCollection for speed
            traj_2d = traj[:, :, :2]
            n_time = traj_2d.shape[1]
            color_vals = cmap(np.linspace(0, 1, n_time))
            segments = []
            seg_colors = []
            for j in range(traj_2d.shape[0]):
                pts = traj_2d[j]  # [T, 2]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                segments.append(segs)
                seg_colors.append(color_vals[:-1])
            segments = np.concatenate(segments, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=0.8)
            ax.add_collection(lc)
            
            # Start and end points
            ax.scatter(traj_2d[:, 0, 0], traj_2d[:, 0, 1], 
                       c='#05009E', s=30, marker='o', label='Trajectory Start', 
                       zorder=5, edgecolors='white', linewidth=1)
            ax.scatter(traj_2d[:, -1, 0], traj_2d[:, -1, 1], 
                       c=c_end, s=30, marker='o', label='Trajectory End', 
                       zorder=5, edgecolors='white', linewidth=1)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("PC1", fontsize=12)
            ax.set_ylabel("PC2", fontsize=12)
            ax.set_title(f"{branch_names[i]}: Trajectories with Timepoint Background", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=12, frameon=False)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{self.args.data_name}_branch{i+1}.png', dpi=300)
            plt.close()

    def _plot_mouse_combined(self, all_trajs, timepoint_data, save_dir, cmap1, cmap2):
        """Plot all branches together."""
        n_branches = len(all_trajs)
        branch_names = [f'Branch {i+1}' for i in range(n_branches)]
        branch_colors = ['#B83CFF', '#50B2D7'][:n_branches]
        
        # Build timepoint key/color/label lists depending on branching
        if 't2_1' in timepoint_data:
            tp_keys = ['t0', 't1', 't2_1', 't2_2']
            tp_colors = ['#05009E', '#A19EFF', '#B83CFF', '#50B2D7']
            tp_labels = ['t=0', 't=1', 't=2 (branch 1)', 't=2 (branch 2)']
        else:
            tp_keys = ['t0', 't1', 't2']
            tp_colors = ['#05009E', '#A19EFF', '#B83CFF']
            tp_labels = ['t=0', 't=1', 't=2']
        
        # Stack list-of-tensors into [B, T, D] numpy arrays
        all_trajs_np = []
        for traj in all_trajs:
            if isinstance(traj, list):
                traj = torch.stack(traj, dim=1)
            if torch.is_tensor(traj):
                traj = traj.cpu().detach().numpy()
            all_trajs_np.append(traj)
        all_trajs = all_trajs_np
        
        # Move timepoint data to numpy
        for key in list(timepoint_data.keys()):
            if torch.is_tensor(timepoint_data[key]):
                timepoint_data[key] = timepoint_data[key].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot timepoint background
        for idx, (t_key, color, label) in enumerate(zip(
            tp_keys, 
            tp_colors, 
            tp_labels
        )):
            if t_key in timepoint_data:
                coords = timepoint_data[t_key]
                ax.scatter(coords[:, 0], coords[:, 1], 
                           c=color, s=80, alpha=0.4, marker='x', 
                           label=f'{label} cells', linewidth=1.5)
        
        # Plot trajectories with color gradients
        cmaps = [cmap1, cmap2]
        for i, traj in enumerate(all_trajs):
            traj_2d = traj[:, :, :2]
            c_end = branch_colors[i]
            cmap = cmaps[i]
            n_time = traj_2d.shape[1]
            color_vals = cmap(np.linspace(0, 1, n_time))
            segments = []
            seg_colors = []
            for j in range(traj_2d.shape[0]):
                pts = traj_2d[j]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                segments.append(segs)
                seg_colors.append(color_vals[:-1])
            segments = np.concatenate(segments, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=0.8)
            ax.add_collection(lc)
            
            ax.scatter(traj_2d[:, 0, 0], traj_2d[:, 0, 1], 
                       c='#05009E', s=30, marker='o', 
                       label=f'{branch_names[i]} Start', 
                       zorder=5, edgecolors='white', linewidth=1)
            ax.scatter(traj_2d[:, -1, 0], traj_2d[:, -1, 1], 
                       c=c_end, s=30, marker='o', 
                       label=f'{branch_names[i]} End', 
                       zorder=5, edgecolors='white', linewidth=1)
        
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)
        ax.set_title("All Branch Trajectories with Timepoint Background", 
                     fontsize=16, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.args.data_name}_combined.png', dpi=300)
        plt.close()


class FlowNetTestClonidine(BranchFlowNetTrainBase):
    """Test class for Clonidine perturbation experiment (1 or 2 branches)."""
    
    def test_step(self, batch, batch_idx):
        # Handle both dict and tuple batch formats from CombinedLoader
        if isinstance(batch, dict) and "test_samples" in batch:
            # New format: {"test_samples": {...}, "metric_samples": {...}}
            main_batch = batch["test_samples"]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
            # Old format with nested structure
            test_samples = batch[0]
            if isinstance(test_samples, dict) and "test_samples" in test_samples:
                main_batch = test_samples["test_samples"][0]
            else:
                main_batch = test_samples
        else:
            # Fallback
            main_batch = batch
        
        # Get timepoint data
        timepoint_data = self.trainer.datamodule.get_timepoint_data()
        device = main_batch["x0"][0].device
        
        # Use val x0 as initial conditions
        x0 = self.trainer.datamodule.val_dataloaders["x0"].dataset.tensors[0].to(device)
        t_span = torch.linspace(0, 1, 100).to(device)
        
        # Define color schemes for clonidine (2 branches)
        custom_colors_1 = ["#05009E", "#A19EFF", "#B83CFF"]
        custom_colors_2 = ["#05009E", "#A19EFF", "#50B2D7"]
        custom_cmap_1 = LinearSegmentedColormap.from_list("cmap1", custom_colors_1)
        custom_cmap_2 = LinearSegmentedColormap.from_list("cmap2", custom_colors_2)
        
        all_trajs = []
        all_endpoints = []

        for i, flow_net in enumerate(self.flow_nets):
            node = NeuralODE(
                flow_model_torch_wrapper(flow_net),
                solver="euler",
                sensitivity="adjoint",
            )

            with torch.no_grad():
                traj = node.trajectory(x0, t_span).cpu()  # [T, B, D]

            traj = torch.transpose(traj, 0, 1)  # [B, T, D]
            all_trajs.append(traj)
            all_endpoints.append(traj[:, -1, :])

        # Run 5 trials with random subsampling for robust metrics
        n_trials = 5
        n_branches = len(self.flow_nets)

        # Gather per-branch ground truth
        gt_data_per_branch = []
        for i in range(n_branches):
            if n_branches == 1:
                key = 't1'
            else:
                key = f't1_{i+1}' if f't1_{i+1}' in timepoint_data else 't1'
            gt_data_per_branch.append(torch.tensor(timepoint_data[key], dtype=torch.float32))
        gt_all = torch.cat(gt_data_per_branch, dim=0)

        # Per-branch metrics (5 trials)
        metrics_dict = {}
        for i in range(n_branches):
            w1_br, w2_br, mmd_br = [], [], []
            pred = all_endpoints[i]
            gt = gt_data_per_branch[i]
            for trial in range(n_trials):
                n_min = min(pred.shape[0], gt.shape[0])
                perm_pred = torch.randperm(pred.shape[0])[:n_min]
                perm_gt = torch.randperm(gt.shape[0])[:n_min]
                m = compute_distribution_distances(pred[perm_pred, :2], gt[perm_gt, :2])
                w1_br.append(m["W1"]); w2_br.append(m["W2"]); mmd_br.append(m["MMD"])
            metrics_dict[f"branch_{i+1}"] = {
                "W1_mean": float(np.mean(w1_br)), "W1_std": float(np.std(w1_br, ddof=1)),
                "W2_mean": float(np.mean(w2_br)), "W2_std": float(np.std(w2_br, ddof=1)),
                "MMD_mean": float(np.mean(mmd_br)), "MMD_std": float(np.std(mmd_br, ddof=1)),
            }
            self.log(f"test/W1_branch{i+1}", np.mean(w1_br), on_epoch=True)
            print(f"Branch {i+1} — W1: {np.mean(w1_br):.6f}±{np.std(w1_br, ddof=1):.6f}, "
                  f"W2: {np.mean(w2_br):.6f}±{np.std(w2_br, ddof=1):.6f}, "
                  f"MMD: {np.mean(mmd_br):.6f}±{np.std(mmd_br, ddof=1):.6f}")

        # Combined metrics (5 trials)
        pred_all = torch.cat(all_endpoints, dim=0)
        w1_trials, w2_trials, mmd_trials = [], [], []
        for trial in range(n_trials):
            n_min = min(pred_all.shape[0], gt_all.shape[0])
            perm_pred = torch.randperm(pred_all.shape[0])[:n_min]
            perm_gt = torch.randperm(gt_all.shape[0])[:n_min]
            m = compute_distribution_distances(pred_all[perm_pred, :2], gt_all[perm_gt, :2])
            w1_trials.append(m["W1"]); w2_trials.append(m["W2"]); mmd_trials.append(m["MMD"])

        w1_mean, w1_std = np.mean(w1_trials), np.std(w1_trials, ddof=1)
        w2_mean, w2_std = np.mean(w2_trials), np.std(w2_trials, ddof=1)
        mmd_mean, mmd_std = np.mean(mmd_trials), np.std(mmd_trials, ddof=1)
        self.log("test/W1_t1_combined", w1_mean, on_epoch=True)
        self.log("test/W2_t1_combined", w2_mean, on_epoch=True)
        self.log("test/MMD_t1_combined", mmd_mean, on_epoch=True)
        metrics_dict['t1_combined'] = {
            "W1_mean": float(w1_mean), "W1_std": float(w1_std),
            "W2_mean": float(w2_mean), "W2_std": float(w2_std),
            "MMD_mean": float(mmd_mean), "MMD_std": float(mmd_std),
            "n_trials": n_trials,
        }
        print(f"\n=== Combined @ t1 ===")
        print(f"W1: {w1_mean:.6f} ± {w1_std:.6f}")
        print(f"W2: {w2_mean:.6f} ± {w2_std:.6f}")
        print(f"MMD: {mmd_mean:.6f} ± {mmd_std:.6f}")

        # Create results directory structure
        run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
        results_dir = os.path.join(self.args.working_dir, 'results', run_name)
        figures_dir = f'{results_dir}/figures'
        os.makedirs(figures_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = f'{results_dir}/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        # Save detailed metrics to CSV
        detailed_csv_path = f'{results_dir}/metrics_detailed.csv'
        with open(detailed_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric_Group', 'W1_Mean', 'W1_Std', 'W2_Mean', 'W2_Std', 'MMD_Mean', 'MMD_Std'])
            for key in sorted(metrics_dict.keys()):
                m = metrics_dict[key]
                writer.writerow([key,
                    f'{m.get("W1_mean", m.get("W1", 0)):.6f}', f'{m.get("W1_std", 0):.6f}',
                    f'{m.get("W2_mean", m.get("W2", 0)):.6f}', f'{m.get("W2_std", 0):.6f}',
                    f'{m.get("MMD_mean", m.get("MMD", 0)):.6f}', f'{m.get("MMD_std", 0):.6f}'])
        print(f"Detailed metrics CSV saved to {detailed_csv_path}")

        # ===== Plot branches =====
        self._plot_clonidine_branches(all_trajs, timepoint_data, figures_dir, custom_cmap_1, custom_cmap_2)
        self._plot_clonidine_combined(all_trajs, timepoint_data, figures_dir)

        print(f"Clonidine figures saved to {figures_dir}")

    def _plot_clonidine_branches(self, all_trajs, timepoint_data, save_dir, cmap1, cmap2):
        """Plot each branch separately."""
        branch_names = ['Branch 1', 'Branch 2']
        branch_colors = ['#B83CFF', '#50B2D7']
        cmaps = [cmap1, cmap2]
        
        # Compute global axis limits – handle single vs multi branch keys
        all_coords = []
        if 't1_1' in timepoint_data:
            tp_keys = ['t0'] + [f't1_{i+1}' for i in range(len(all_trajs))]
        else:
            tp_keys = ['t0', 't1']
        for key in tp_keys:
            all_coords.append(timepoint_data[key][:, :2])
        for traj in all_trajs:
            all_coords.append(traj.reshape(-1, traj.shape[-1])[:, :2])
        
        all_coords = np.concatenate(all_coords, axis=0)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        for i, traj in enumerate(all_trajs):
            fig, ax = plt.subplots(figsize=(10, 8))
            c_end = branch_colors[i]
            
            # Plot timepoint background
            t1_key = f't1_{i+1}' if f't1_{i+1}' in timepoint_data else 't1'
            coords_list = [timepoint_data['t0'], timepoint_data[t1_key]]
            tp_colors = ['#05009E', c_end]
            t1_label = f"t=1 (branch {i+1})" if len(all_trajs) > 1 else "t=1"
            tp_labels = ["t=0", t1_label]
            
            for coords, color, label in zip(coords_list, tp_colors, tp_labels):
                ax.scatter(coords[:, 0], coords[:, 1], 
                           c=color, s=80, alpha=0.4, marker='x', 
                           label=f'{label} cells', linewidth=1.5)
            
            # Plot continuous trajectories with LineCollection for speed
            traj_2d = traj[:, :, :2]
            n_time = traj_2d.shape[1]
            color_vals = cmaps[i](np.linspace(0, 1, n_time))
            segments = []
            seg_colors = []
            for j in range(traj_2d.shape[0]):
                pts = traj_2d[j]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                segments.append(segs)
                seg_colors.append(color_vals[:-1])
            segments = np.concatenate(segments, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=0.8)
            ax.add_collection(lc)
            
            # Start and end points
            ax.scatter(traj_2d[:, 0, 0], traj_2d[:, 0, 1], 
                       c='#05009E', s=30, marker='o', label='Trajectory Start', 
                       zorder=5, edgecolors='white', linewidth=1)
            ax.scatter(traj_2d[:, -1, 0], traj_2d[:, -1, 1], 
                       c=c_end, s=30, marker='o', label='Trajectory End', 
                       zorder=5, edgecolors='white', linewidth=1)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("PC1", fontsize=12)
            ax.set_ylabel("PC2", fontsize=12)
            ax.set_title(f"{branch_names[i]}: Trajectories with Timepoint Background", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=16, frameon=False)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{self.args.data_name}_branch{i+1}.png', dpi=300)
            plt.close()

    def _plot_clonidine_combined(self, all_trajs, timepoint_data, save_dir):
        """Plot all branches together."""
        branch_names = ['Branch 1', 'Branch 2']
        branch_colors = ['#B83CFF', '#50B2D7']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Build timepoint keys/colors/labels depending on single vs multi branch
        if 't1_1' in timepoint_data:
            tp_keys = ['t0'] + [f't1_{j+1}' for j in range(len(all_trajs))]
            tp_labels_list = ['t=0'] + [f't=1 (branch {j+1})' for j in range(len(all_trajs))]
        else:
            tp_keys = ['t0', 't1']
            tp_labels_list = ['t=0', 't=1']
        tp_colors = ['#05009E', '#B83CFF', '#50B2D7'][:len(tp_keys)]
        
        # Plot timepoint background
        for t_key, color, label in zip(tp_keys, tp_colors, tp_labels_list):
            coords = timepoint_data[t_key]
            ax.scatter(coords[:, 0], coords[:, 1], 
                       c=color, s=80, alpha=0.4, marker='x', 
                       label=f'{label} cells', linewidth=1.5)
        
        # Plot trajectories with color gradients
        custom_colors_1 = ["#05009E", "#A19EFF", "#B83CFF"]
        custom_colors_2 = ["#05009E", "#A19EFF", "#50B2D7"]
        cmaps = [
            LinearSegmentedColormap.from_list("clon_cmap1", custom_colors_1),
            LinearSegmentedColormap.from_list("clon_cmap2", custom_colors_2),
        ]
        for i, traj in enumerate(all_trajs):
            traj_2d = traj[:, :, :2]
            c_end = branch_colors[i]
            cmap = cmaps[i]
            n_time = traj_2d.shape[1]
            color_vals = cmap(np.linspace(0, 1, n_time))
            segments = []
            seg_colors = []
            for j in range(traj_2d.shape[0]):
                pts = traj_2d[j]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                segments.append(segs)
                seg_colors.append(color_vals[:-1])
            segments = np.concatenate(segments, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=0.8)
            ax.add_collection(lc)
            
            ax.scatter(traj_2d[:, 0, 0], traj_2d[:, 0, 1], 
                       c='#05009E', s=30, marker='o', 
                       label=f'{branch_names[i]} Start', 
                       zorder=5, edgecolors='white', linewidth=1)
            ax.scatter(traj_2d[:, -1, 0], traj_2d[:, -1, 1], 
                       c=c_end, s=30, marker='o', 
                       label=f'{branch_names[i]} End', 
                       zorder=5, edgecolors='white', linewidth=1)
        
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)
        ax.set_title("All Branch Trajectories with Timepoint Background", 
                     fontsize=16, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.args.data_name}_combined.png', dpi=300)
        plt.close()


class FlowNetTestTrametinib(BranchFlowNetTrainBase):
    """Test class for Trametinib perturbation experiment (1 or 3 branches)."""
    
    def test_step(self, batch, batch_idx):
        # Handle both dict and tuple batch formats from CombinedLoader
        if isinstance(batch, dict) and "test_samples" in batch:
            # New format: {"test_samples": {...}, "metric_samples": {...}}
            main_batch = batch["test_samples"]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
            # Old format with nested structure
            test_samples = batch[0]
            if isinstance(test_samples, dict) and "test_samples" in test_samples:
                main_batch = test_samples["test_samples"][0]
            else:
                main_batch = test_samples
        else:
            # Fallback
            main_batch = batch
        
        # Get timepoint data
        timepoint_data = self.trainer.datamodule.get_timepoint_data()
        device = main_batch["x0"][0].device
        
        # Use val x0 as initial conditions
        x0 = self.trainer.datamodule.val_dataloaders["x0"].dataset.tensors[0].to(device)
        t_span = torch.linspace(0, 1, 100).to(device)
        
        # Define color schemes for trametinib (3 branches)
        custom_colors_1 = ["#05009E", "#A19EFF", "#9793F8"]
        custom_colors_2 = ["#05009E", "#A19EFF", "#50B2D7"]
        custom_colors_3 = ["#05009E", "#A19EFF", "#B83CFF"]
        custom_cmap_1 = LinearSegmentedColormap.from_list("cmap1", custom_colors_1)
        custom_cmap_2 = LinearSegmentedColormap.from_list("cmap2", custom_colors_2)
        custom_cmap_3 = LinearSegmentedColormap.from_list("cmap3", custom_colors_3)
        
        all_trajs = []
        all_endpoints = []

        for i, flow_net in enumerate(self.flow_nets):
            node = NeuralODE(
                flow_model_torch_wrapper(flow_net),
                solver="euler",
                sensitivity="adjoint",
            )

            with torch.no_grad():
                traj = node.trajectory(x0, t_span).cpu()  # [T, B, D]

            traj = torch.transpose(traj, 0, 1)  # [B, T, D]
            all_trajs.append(traj)
            all_endpoints.append(traj[:, -1, :])

        # Run 5 trials with random subsampling for robust metrics
        n_trials = 5
        n_branches = len(self.flow_nets)

        # Gather per-branch ground truth
        gt_data_per_branch = []
        for i in range(n_branches):
            if n_branches == 1:
                key = 't1'
            else:
                key = f't1_{i+1}' if f't1_{i+1}' in timepoint_data else 't1'
            gt_data_per_branch.append(torch.tensor(timepoint_data[key], dtype=torch.float32))
        gt_all = torch.cat(gt_data_per_branch, dim=0)

        # Per-branch metrics (5 trials)
        metrics_dict = {}
        for i in range(n_branches):
            w1_br, w2_br, mmd_br = [], [], []
            pred = all_endpoints[i]
            gt = gt_data_per_branch[i]
            for trial in range(n_trials):
                n_min = min(pred.shape[0], gt.shape[0])
                perm_pred = torch.randperm(pred.shape[0])[:n_min]
                perm_gt = torch.randperm(gt.shape[0])[:n_min]
                m = compute_distribution_distances(pred[perm_pred, :2], gt[perm_gt, :2])
                w1_br.append(m["W1"]); w2_br.append(m["W2"]); mmd_br.append(m["MMD"])
            metrics_dict[f"branch_{i+1}"] = {
                "W1_mean": float(np.mean(w1_br)), "W1_std": float(np.std(w1_br, ddof=1)),
                "W2_mean": float(np.mean(w2_br)), "W2_std": float(np.std(w2_br, ddof=1)),
                "MMD_mean": float(np.mean(mmd_br)), "MMD_std": float(np.std(mmd_br, ddof=1)),
            }
            self.log(f"test/W1_branch{i+1}", np.mean(w1_br), on_epoch=True)
            print(f"Branch {i+1} — W1: {np.mean(w1_br):.6f}±{np.std(w1_br, ddof=1):.6f}, "
                  f"W2: {np.mean(w2_br):.6f}±{np.std(w2_br, ddof=1):.6f}, "
                  f"MMD: {np.mean(mmd_br):.6f}±{np.std(mmd_br, ddof=1):.6f}")

        # Combined metrics (5 trials)
        pred_all = torch.cat(all_endpoints, dim=0)
        w1_trials, w2_trials, mmd_trials = [], [], []
        for trial in range(n_trials):
            n_min = min(pred_all.shape[0], gt_all.shape[0])
            perm_pred = torch.randperm(pred_all.shape[0])[:n_min]
            perm_gt = torch.randperm(gt_all.shape[0])[:n_min]
            m = compute_distribution_distances(pred_all[perm_pred, :2], gt_all[perm_gt, :2])
            w1_trials.append(m["W1"]); w2_trials.append(m["W2"]); mmd_trials.append(m["MMD"])

        w1_mean, w1_std = np.mean(w1_trials), np.std(w1_trials, ddof=1)
        w2_mean, w2_std = np.mean(w2_trials), np.std(w2_trials, ddof=1)
        mmd_mean, mmd_std = np.mean(mmd_trials), np.std(mmd_trials, ddof=1)
        self.log("test/W1_t1_combined", w1_mean, on_epoch=True)
        self.log("test/W2_t1_combined", w2_mean, on_epoch=True)
        self.log("test/MMD_t1_combined", mmd_mean, on_epoch=True)
        metrics_dict['t1_combined'] = {
            "W1_mean": float(w1_mean), "W1_std": float(w1_std),
            "W2_mean": float(w2_mean), "W2_std": float(w2_std),
            "MMD_mean": float(mmd_mean), "MMD_std": float(mmd_std),
            "n_trials": n_trials,
        }
        print(f"\n=== Combined @ t1 ===")
        print(f"W1: {w1_mean:.6f} ± {w1_std:.6f}")
        print(f"W2: {w2_mean:.6f} ± {w2_std:.6f}")
        print(f"MMD: {mmd_mean:.6f} ± {mmd_std:.6f}")

        # Create results directory structure
        run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
        results_dir = os.path.join(self.args.working_dir, 'results', run_name)
        figures_dir = f'{results_dir}/figures'
        os.makedirs(figures_dir, exist_ok=True)
        
        # Save metrics to JSON
        metrics_path = f'{results_dir}/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        # Save detailed metrics to CSV
        detailed_csv_path = f'{results_dir}/metrics_detailed.csv'
        with open(detailed_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric_Group', 'W1_Mean', 'W1_Std', 'W2_Mean', 'W2_Std', 'MMD_Mean', 'MMD_Std'])
            for key in sorted(metrics_dict.keys()):
                m = metrics_dict[key]
                writer.writerow([key,
                    f'{m.get("W1_mean", m.get("W1", 0)):.6f}', f'{m.get("W1_std", 0):.6f}',
                    f'{m.get("W2_mean", m.get("W2", 0)):.6f}', f'{m.get("W2_std", 0):.6f}',
                    f'{m.get("MMD_mean", m.get("MMD", 0)):.6f}', f'{m.get("MMD_std", 0):.6f}'])
        print(f"Detailed metrics CSV saved to {detailed_csv_path}")

        # ===== Plot branches =====
        self._plot_trametinib_branches(all_trajs, timepoint_data, figures_dir, 
                                       custom_cmap_1, custom_cmap_2, custom_cmap_3)
        self._plot_trametinib_combined(all_trajs, timepoint_data, figures_dir)

        print(f"Trametinib figures saved to {figures_dir}")

    def _plot_trametinib_branches(self, all_trajs, timepoint_data, save_dir, 
                                   cmap1, cmap2, cmap3):
        """Plot each branch separately."""
        branch_names = ['Branch 1', 'Branch 2', 'Branch 3']
        branch_colors = ['#9793F8', '#50B2D7', '#B83CFF']
        cmaps = [cmap1, cmap2, cmap3]
        
        # Compute global axis limits – handle single vs multi branch keys
        all_coords = []
        if 't1_1' in timepoint_data:
            tp_keys = ['t0'] + [f't1_{i+1}' for i in range(len(all_trajs))]
        else:
            tp_keys = ['t0', 't1']
        for key in tp_keys:
            all_coords.append(timepoint_data[key][:, :2])
        for traj in all_trajs:
            all_coords.append(traj.reshape(-1, traj.shape[-1])[:, :2])
        
        all_coords = np.concatenate(all_coords, axis=0)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        for i, traj in enumerate(all_trajs):
            fig, ax = plt.subplots(figsize=(10, 8))
            c_end = branch_colors[i]
            
            # Plot timepoint background
            t1_key = f't1_{i+1}' if f't1_{i+1}' in timepoint_data else 't1'
            coords_list = [timepoint_data['t0'], timepoint_data[t1_key]]
            tp_colors = ['#05009E', c_end]
            t1_label = f"t=1 (branch {i+1})" if len(all_trajs) > 1 else "t=1"
            tp_labels = ["t=0", t1_label]
            
            for coords, color, label in zip(coords_list, tp_colors, tp_labels):
                ax.scatter(coords[:, 0], coords[:, 1], 
                           c=color, s=80, alpha=0.4, marker='x', 
                           label=f'{label} cells', linewidth=1.5)
            
            # Plot continuous trajectories with LineCollection for speed
            traj_2d = traj[:, :, :2]
            n_time = traj_2d.shape[1]
            color_vals = cmaps[i](np.linspace(0, 1, n_time))
            segments = []
            seg_colors = []
            for j in range(traj_2d.shape[0]):
                pts = traj_2d[j]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                segments.append(segs)
                seg_colors.append(color_vals[:-1])
            segments = np.concatenate(segments, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=0.8)
            ax.add_collection(lc)
            
            # Start and end points
            ax.scatter(traj_2d[:, 0, 0], traj_2d[:, 0, 1], 
                       c='#05009E', s=30, marker='o', label='Trajectory Start', 
                       zorder=5, edgecolors='white', linewidth=1)
            ax.scatter(traj_2d[:, -1, 0], traj_2d[:, -1, 1], 
                       c=c_end, s=30, marker='o', label='Trajectory End', 
                       zorder=5, edgecolors='white', linewidth=1)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("PC1", fontsize=12)
            ax.set_ylabel("PC2", fontsize=12)
            ax.set_title(f"{branch_names[i]}: Trajectories with Timepoint Background", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=16, frameon=False)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{self.args.data_name}_branch{i+1}.png', dpi=300)
            plt.close()

    def _plot_trametinib_combined(self, all_trajs, timepoint_data, save_dir):
        """Plot all 3 branches together."""
        branch_names = ['Branch 1', 'Branch 2', 'Branch 3']
        branch_colors = ['#9793F8', '#50B2D7', '#B83CFF']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Build timepoint keys/colors/labels depending on single vs multi branch
        if 't1_1' in timepoint_data:
            tp_keys = ['t0'] + [f't1_{j+1}' for j in range(len(all_trajs))]
            tp_labels_list = ['t=0'] + [f't=1 (branch {j+1})' for j in range(len(all_trajs))]
        else:
            tp_keys = ['t0', 't1']
            tp_labels_list = ['t=0', 't=1']
        tp_colors = ['#05009E', '#9793F8', '#50B2D7', '#B83CFF'][:len(tp_keys)]
        
        # Plot timepoint background
        for t_key, color, label in zip(tp_keys, tp_colors, tp_labels_list):
            coords = timepoint_data[t_key]
            ax.scatter(coords[:, 0], coords[:, 1], 
                       c=color, s=80, alpha=0.4, marker='x', 
                       label=f'{label} cells', linewidth=1.5)
        
        # Plot trajectories with color gradients
        custom_colors_1 = ["#05009E", "#A19EFF", "#9793F8"]
        custom_colors_2 = ["#05009E", "#A19EFF", "#50B2D7"]
        custom_colors_3 = ["#05009E", "#A19EFF", "#D577FF"]
        cmaps = [
            LinearSegmentedColormap.from_list("tram_cmap1", custom_colors_1),
            LinearSegmentedColormap.from_list("tram_cmap2", custom_colors_2),
            LinearSegmentedColormap.from_list("tram_cmap3", custom_colors_3),
        ]
        for i, traj in enumerate(all_trajs):
            traj_2d = traj[:, :, :2]
            c_end = branch_colors[i]
            cmap = cmaps[i]
            n_time = traj_2d.shape[1]
            color_vals = cmap(np.linspace(0, 1, n_time))
            segments = []
            seg_colors = []
            for j in range(traj_2d.shape[0]):
                pts = traj_2d[j]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                segments.append(segs)
                seg_colors.append(color_vals[:-1])
            segments = np.concatenate(segments, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=0.8)
            ax.add_collection(lc)
            
            ax.scatter(traj_2d[:, 0, 0], traj_2d[:, 0, 1], 
                       c='#05009E', s=30, marker='o', 
                       label=f'{branch_names[i]} Start', 
                       zorder=5, edgecolors='white', linewidth=1)
            ax.scatter(traj_2d[:, -1, 0], traj_2d[:, -1, 1], 
                       c=c_end, s=30, marker='o', 
                       label=f'{branch_names[i]} End', 
                       zorder=5, edgecolors='white', linewidth=1)
        
        ax.set_xlabel("PC1", fontsize=14)
        ax.set_ylabel("PC2", fontsize=14)
        ax.set_title("All Branch Trajectories with Timepoint Background", 
                     fontsize=16, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12, frameon=False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.args.data_name}_combined.png', dpi=300)
        plt.close()

class FlowNetTestVeres(GrowthNetTrain):
    """Test class for Veres pancreatic endocrinogenesis experiment (3 or 5 branches)."""
    
    def test_step(self, batch, batch_idx):
        # Handle both tuple and dict batch formats from CombinedLoader
        if isinstance(batch, dict):
            main_batch = batch["test_samples"][0]
            metric_batch = batch["metric_samples"][0]
        else:
            # batch is a list/tuple
            if isinstance(batch[0], dict):
                # batch[0] contains the dict with test_samples and metric_samples
                main_batch = batch[0]["test_samples"][0]
                metric_batch = batch[0]["metric_samples"][0]
            else:
                # batch is a tuple: (test_samples, metric_samples)
                main_batch = batch[0][0]
                metric_batch = batch[1][0]
        
        # Get timepoint data (full datasets, not just val split)
        timepoint_data = self.trainer.datamodule.get_timepoint_data()
        device = main_batch["x0"][0].device
        
        # Use val x0 as initial conditions
        x0_all = self.trainer.datamodule.val_dataloaders["x0"].dataset.tensors[0].to(device)
        w0_all = torch.ones(x0_all.shape[0], 1, dtype=torch.float32).to(device)
        full_batch = {"x0": (x0_all, w0_all)}
        
        time_points, all_endpoints, all_trajs, mass_over_time, energy_over_time, weights_over_time = self.get_mass_and_position(full_batch, metric_batch)

        n_branches = len(self.flow_nets)
        
        # trajectory time grid
        t_span = torch.linspace(0, 1, 101).to(device)

        # `all_trajs` returned from `get_mass_and_position` is expected to be a list where each
        # element is a sequence of per-timepoint tensors for that branch (shape [B, D] each).
        # Convert each branch to [T, B, D] then to [B, T, D] for downstream processing.
        trajs_TBD = [torch.stack(branch_list, dim=0) for branch_list in all_trajs]  # each is [T, B, D]
        trajs_BTD = [t.permute(1, 0, 2) for t in trajs_TBD]  # each -> [B, T, D]

        all_trajs = []
        all_endpoints = []
        # will store per-branch intermediate frames: each entry -> tensor [B, n_intermediate, D]
        all_intermediates = []

        for traj in trajs_BTD:
            # traj is [B, T, D]
            # optionally inverse-transform if whitened
            if self.whiten:
                traj_np = traj.detach().cpu().numpy()
                n_samples, n_time, n_dims = traj_np.shape
                traj_flat = traj_np.reshape(-1, n_dims)
                traj_inv_flat = self.trainer.datamodule.scaler.inverse_transform(traj_flat)
                traj_inv = traj_inv_flat.reshape(n_samples, n_time, n_dims)
                traj = torch.tensor(traj_inv, dtype=torch.float32)

            all_trajs.append(traj)

            # Collect six evenly spaced intermediate frames between t=0 and t=1 (exclude endpoints)
            n_T = traj.shape[1]
            # choose 8 points including endpoints -> take inner 6 as intermediates
            inter_times = np.linspace(0.0, 1.0, 8)[1:-1]  # 6 values
            inter_indices = [int(round(t * (n_T - 1))) for t in inter_times]
            # stack per-branch intermediate frames -> [B, 6, D]
            intermediates = torch.stack([traj[:, idx, :] for idx in inter_indices], dim=1)
            all_intermediates.append(intermediates)

            # Final endpoints (t=1)
            all_endpoints.append(traj[:, -1, :])

        # Run 5 trials with random subsampling for robust metrics
        n_trials = 5
        metrics_dict = {}

        # --- Intermediate timepoints (t1-t6) combined metrics ---
        intermediate_keys = sorted([k for k in timepoint_data.keys()
                                   if k.startswith('t') and '_' not in k and k != 't0'])

        if intermediate_keys:
            n_evals = min(6, len(intermediate_keys))
            for j in range(n_evals):
                intermediate_key = intermediate_keys[j]
                true_data_intermediate = torch.tensor(timepoint_data[intermediate_key], dtype=torch.float32)

                # Gather predicted intermediates across all branches
                raw_intermediates = [branch[:, j, :] for branch in all_intermediates]
                all_raw_concat = torch.cat(raw_intermediates, dim=0).cpu()  # [n_branches*B, D]

                w1_t, w2_t, mmd_t = [], [], []
                w1_t_full, w2_t_full, mmd_t_full = [], [], []
                for trial in range(n_trials):
                    n_min = min(all_raw_concat.shape[0], true_data_intermediate.shape[0])
                    perm_pred = torch.randperm(all_raw_concat.shape[0])[:n_min]
                    perm_gt = torch.randperm(true_data_intermediate.shape[0])[:n_min]
                    # 2D metrics (PC1-PC2)
                    m = compute_distribution_distances(
                        all_raw_concat[perm_pred, :2], true_data_intermediate[perm_gt, :2])
                    w1_t.append(m["W1"]); w2_t.append(m["W2"]); mmd_t.append(m["MMD"])
                    # Full-dimensional metrics (all PCs)
                    m_full = compute_distribution_distances(
                        all_raw_concat[perm_pred], true_data_intermediate[perm_gt])
                    w1_t_full.append(m_full["W1"]); w2_t_full.append(m_full["W2"]); mmd_t_full.append(m_full["MMD"])

                metrics_dict[f'{intermediate_key}_combined'] = {
                    "W1_mean": float(np.mean(w1_t)), "W1_std": float(np.std(w1_t, ddof=1)),
                    "W2_mean": float(np.mean(w2_t)), "W2_std": float(np.std(w2_t, ddof=1)),
                    "MMD_mean": float(np.mean(mmd_t)), "MMD_std": float(np.std(mmd_t, ddof=1)),
                    "W1_full_mean": float(np.mean(w1_t_full)), "W1_full_std": float(np.std(w1_t_full, ddof=1)),
                    "W2_full_mean": float(np.mean(w2_t_full)), "W2_full_std": float(np.std(w2_t_full, ddof=1)),
                    "MMD_full_mean": float(np.mean(mmd_t_full)), "MMD_full_std": float(np.std(mmd_t_full, ddof=1)),
                }
                self.log(f"test/W1_{intermediate_key}_combined", np.mean(w1_t), on_epoch=True)
                self.log(f"test/W1_full_{intermediate_key}_combined", np.mean(w1_t_full), on_epoch=True)
                print(f"{intermediate_key} combined — W1: {np.mean(w1_t):.6f}±{np.std(w1_t, ddof=1):.6f}, "
                      f"W2: {np.mean(w2_t):.6f}±{np.std(w2_t, ddof=1):.6f}, "
                      f"MMD: {np.mean(mmd_t):.6f}±{np.std(mmd_t, ddof=1):.6f}")
                print(f"{intermediate_key} combined (full) — W1: {np.mean(w1_t_full):.6f}±{np.std(w1_t_full, ddof=1):.6f}, "
                      f"W2: {np.mean(w2_t_full):.6f}±{np.std(w2_t_full, ddof=1):.6f}, "
                      f"MMD: {np.mean(mmd_t_full):.6f}±{np.std(mmd_t_full, ddof=1):.6f}")

        # --- Final timepoint per-branch metrics ---
        gt_keys = sorted([k for k in timepoint_data.keys() if k.startswith('t7_')])
        for i, endpoints in enumerate(all_endpoints):
            true_data_key = f"t7_{i}"
            if true_data_key not in timepoint_data:
                print(f"Warning: {true_data_key} not found in timepoint_data")
                continue
            gt = torch.tensor(timepoint_data[true_data_key], dtype=torch.float32)
            pred = endpoints.cpu()

            w1_br, w2_br, mmd_br = [], [], []
            w1_br_full, w2_br_full, mmd_br_full = [], [], []
            for trial in range(n_trials):
                n_min = min(pred.shape[0], gt.shape[0])
                perm_pred = torch.randperm(pred.shape[0])[:n_min]
                perm_gt = torch.randperm(gt.shape[0])[:n_min]
                # 2D metrics (PC1-PC2)
                m = compute_distribution_distances(pred[perm_pred, :2], gt[perm_gt, :2])
                w1_br.append(m["W1"]); w2_br.append(m["W2"]); mmd_br.append(m["MMD"])
                # Full-dimensional metrics (all PCs)
                m_full = compute_distribution_distances(pred[perm_pred], gt[perm_gt])
                w1_br_full.append(m_full["W1"]); w2_br_full.append(m_full["W2"]); mmd_br_full.append(m_full["MMD"])

            metrics_dict[f"branch_{i}"] = {
                "W1_mean": float(np.mean(w1_br)), "W1_std": float(np.std(w1_br, ddof=1)),
                "W2_mean": float(np.mean(w2_br)), "W2_std": float(np.std(w2_br, ddof=1)),
                "MMD_mean": float(np.mean(mmd_br)), "MMD_std": float(np.std(mmd_br, ddof=1)),
                "W1_full_mean": float(np.mean(w1_br_full)), "W1_full_std": float(np.std(w1_br_full, ddof=1)),
                "W2_full_mean": float(np.mean(w2_br_full)), "W2_full_std": float(np.std(w2_br_full, ddof=1)),
                "MMD_full_mean": float(np.mean(mmd_br_full)), "MMD_full_std": float(np.std(mmd_br_full, ddof=1)),
            }
            self.log(f"test/W1_branch{i}", np.mean(w1_br), on_epoch=True)
            self.log(f"test/W1_full_branch{i}", np.mean(w1_br_full), on_epoch=True)
            print(f"Branch {i} — W1: {np.mean(w1_br):.6f}±{np.std(w1_br, ddof=1):.6f}, "
                  f"W2: {np.mean(w2_br):.6f}±{np.std(w2_br, ddof=1):.6f}, "
                  f"MMD: {np.mean(mmd_br):.6f}±{np.std(mmd_br, ddof=1):.6f}")
            print(f"Branch {i} (full) — W1: {np.mean(w1_br_full):.6f}±{np.std(w1_br_full, ddof=1):.6f}, "
                  f"W2: {np.mean(w2_br_full):.6f}±{np.std(w2_br_full, ddof=1):.6f}, "
                  f"MMD: {np.mean(mmd_br_full):.6f}±{np.std(mmd_br_full, ddof=1):.6f}")

        # --- Final timepoint combined metrics ---
        gt_list = [torch.tensor(timepoint_data[k], dtype=torch.float32) for k in gt_keys]
        if len(gt_list) > 0 and len(all_endpoints) > 0:
            gt_all = torch.cat(gt_list, dim=0)
            pred_all = torch.cat([e.cpu() for e in all_endpoints], dim=0)

            w1_trials, w2_trials, mmd_trials = [], [], []
            w1_trials_full, w2_trials_full, mmd_trials_full = [], [], []
            for trial in range(n_trials):
                n_min = min(pred_all.shape[0], gt_all.shape[0])
                perm_pred = torch.randperm(pred_all.shape[0])[:n_min]
                perm_gt = torch.randperm(gt_all.shape[0])[:n_min]
                # 2D metrics (PC1-PC2)
                m = compute_distribution_distances(pred_all[perm_pred, :2], gt_all[perm_gt, :2])
                w1_trials.append(m["W1"]); w2_trials.append(m["W2"]); mmd_trials.append(m["MMD"])
                # Full-dimensional metrics (all PCs)
                m_full = compute_distribution_distances(pred_all[perm_pred], gt_all[perm_gt])
                w1_trials_full.append(m_full["W1"]); w2_trials_full.append(m_full["W2"]); mmd_trials_full.append(m_full["MMD"])

            w1_mean, w1_std = np.mean(w1_trials), np.std(w1_trials, ddof=1)
            w2_mean, w2_std = np.mean(w2_trials), np.std(w2_trials, ddof=1)
            mmd_mean, mmd_std = np.mean(mmd_trials), np.std(mmd_trials, ddof=1)
            w1_mean_f, w1_std_f = np.mean(w1_trials_full), np.std(w1_trials_full, ddof=1)
            w2_mean_f, w2_std_f = np.mean(w2_trials_full), np.std(w2_trials_full, ddof=1)
            mmd_mean_f, mmd_std_f = np.mean(mmd_trials_full), np.std(mmd_trials_full, ddof=1)
            self.log("test/W1_t7_combined", w1_mean, on_epoch=True)
            self.log("test/W2_t7_combined", w2_mean, on_epoch=True)
            self.log("test/MMD_t7_combined", mmd_mean, on_epoch=True)
            self.log("test/W1_full_t7_combined", w1_mean_f, on_epoch=True)
            self.log("test/W2_full_t7_combined", w2_mean_f, on_epoch=True)
            self.log("test/MMD_full_t7_combined", mmd_mean_f, on_epoch=True)
            metrics_dict['t7_combined'] = {
                "W1_mean": float(w1_mean), "W1_std": float(w1_std),
                "W2_mean": float(w2_mean), "W2_std": float(w2_std),
                "MMD_mean": float(mmd_mean), "MMD_std": float(mmd_std),
                "W1_full_mean": float(w1_mean_f), "W1_full_std": float(w1_std_f),
                "W2_full_mean": float(w2_mean_f), "W2_full_std": float(w2_std_f),
                "MMD_full_mean": float(mmd_mean_f), "MMD_full_std": float(mmd_std_f),
                "n_trials": n_trials,
            }
            print(f"\n=== Combined @ t7 ===")
            print(f"W1: {w1_mean:.6f} ± {w1_std:.6f}")
            print(f"W2: {w2_mean:.6f} ± {w2_std:.6f}")
            print(f"MMD: {mmd_mean:.6f} ± {mmd_std:.6f}")
            print(f"W1 (full): {w1_mean_f:.6f} ± {w1_std_f:.6f}")
            print(f"W2 (full): {w2_mean_f:.6f} ± {w2_std_f:.6f}")
            print(f"MMD (full): {mmd_mean_f:.6f} ± {mmd_std_f:.6f}")

        # Create results directory structure
        run_name = self.args.run_name if hasattr(self.args, 'run_name') and self.args.run_name else self.args.data_name
        results_dir = os.path.join(self.args.working_dir, 'results', run_name)
        figures_dir = f'{results_dir}/figures'
        os.makedirs(figures_dir, exist_ok=True)

        # Save metrics to JSON
        metrics_path = f'{results_dir}/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

        # Save detailed metrics to CSV
        detailed_csv_path = f'{results_dir}/metrics_detailed.csv'
        with open(detailed_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric_Group',
                            'W1_Mean', 'W1_Std', 'W2_Mean', 'W2_Std', 'MMD_Mean', 'MMD_Std',
                            'W1_Full_Mean', 'W1_Full_Std', 'W2_Full_Mean', 'W2_Full_Std', 'MMD_Full_Mean', 'MMD_Full_Std'])
            for key in sorted(metrics_dict.keys()):
                m = metrics_dict[key]
                writer.writerow([key,
                    f'{m.get("W1_mean", 0):.6f}', f'{m.get("W1_std", 0):.6f}',
                    f'{m.get("W2_mean", 0):.6f}', f'{m.get("W2_std", 0):.6f}',
                    f'{m.get("MMD_mean", 0):.6f}', f'{m.get("MMD_std", 0):.6f}',
                    f'{m.get("W1_full_mean", 0):.6f}', f'{m.get("W1_full_std", 0):.6f}',
                    f'{m.get("W2_full_mean", 0):.6f}', f'{m.get("W2_full_std", 0):.6f}',
                    f'{m.get("MMD_full_mean", 0):.6f}', f'{m.get("MMD_full_std", 0):.6f}'])
        print(f"Detailed metrics CSV saved to {detailed_csv_path}")

        # ===== Plot branches =====
        self._plot_veres_branches(all_trajs, timepoint_data, figures_dir, n_branches)
        self._plot_veres_combined(all_trajs, timepoint_data, figures_dir, n_branches)

        print(f"Veres figures saved to {figures_dir}")

    def _plot_veres_branches(self, all_trajs, timepoint_data, save_dir, n_branches):
        """Plot each branch separately in PCA space (PC1 vs PC2)."""
        branch_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9',
                 '#74B9FF', '#A29BFE', '#FFB74D', '#AED581', '#F06292', '#BA68C8',
                 '#4DB6AC', '#81C784', '#FFD54F', '#90A4AE', '#F48FB1', '#CE93D8',
                 '#64B5F6', '#C5E1A5']
        
        # Project to first 2 PCs (data is already in PCA space)
        t0_2d = timepoint_data['t0'].cpu().numpy()[:, :2]
        t7_2d = [timepoint_data[f't7_{i}'].cpu().numpy()[:, :2] for i in range(n_branches)]
        
        # Slice trajectories to first 2 PCs
        trajs_2d = []
        for traj in all_trajs:
            trajs_2d.append(traj.cpu().numpy()[:, :, :2])  # [n_samples, n_time, 2]
        
        # Compute global axis limits
        all_coords = [t0_2d] + t7_2d
        for traj_2d in trajs_2d:
            all_coords.append(traj_2d.reshape(-1, 2))
        
        all_coords = np.concatenate(all_coords, axis=0)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
        
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin

        for i, traj_2d in enumerate(trajs_2d):
            fig, ax = plt.subplots(figsize=(10, 8))
            c_end = branch_colors[i % len(branch_colors)]
            
            # Plot timepoint background
            ax.scatter(t0_2d[:, 0], t0_2d[:, 1], 
                       c='#05009E', s=80, alpha=0.4, marker='x', 
                       label='t=0 cells', linewidth=1.5)
            ax.scatter(t7_2d[i][:, 0], t7_2d[i][:, 1], 
                       c=c_end, s=80, alpha=0.4, marker='x', 
                       label=f't=7 (branch {i+1}) cells', linewidth=1.5)
            
            # Plot continuous trajectories with LineCollection for speed
            cmap_colors = ["#05009E", "#A19EFF", c_end]
            cmap = LinearSegmentedColormap.from_list(f"veres_cmap_{i}", cmap_colors)
            n_time = traj_2d.shape[1]
            segments = []
            seg_colors = []
            color_vals = cmap(np.linspace(0, 1, n_time))
            for j in range(traj_2d.shape[0]):
                pts = traj_2d[j]  # [T, 2]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)  # [T-1, 2, 2]
                segments.append(segs)
                seg_colors.append(color_vals[:-1])
            segments = np.concatenate(segments, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(segments, colors=seg_colors, linewidths=2, alpha=0.8)
            ax.add_collection(lc)
            
            # Start and end points
            ax.scatter(traj_2d[:, 0, 0], traj_2d[:, 0, 1], 
                       c='#05009E', s=30, marker='o', label='Trajectory start (t=0)', 
                       zorder=5, edgecolors='white', linewidth=1)
            ax.scatter(traj_2d[:, -1, 0], traj_2d[:, -1, 1], 
                       c=c_end, s=30, marker='o', label='Trajectory end (t=1)', 
                       zorder=5, edgecolors='white', linewidth=1)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("PC 1", fontsize=12)
            ax.set_ylabel("PC 2", fontsize=12)
            ax.set_title(f"Branch {i+1}: Trajectories (PCA)", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9, frameon=False)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{self.args.data_name}_branch{i+1}.png', dpi=300)
            plt.close()

    def _plot_veres_combined(self, all_trajs, timepoint_data, save_dir, n_branches):
        """Plot all branches together in PCA space (PC1 vs PC2)."""
        branch_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9',
                 '#74B9FF', '#A29BFE', '#FFB74D', '#AED581', '#F06292', '#BA68C8',
                 '#4DB6AC', '#81C784', '#FFD54F', '#90A4AE', '#F48FB1', '#CE93D8',
                 '#64B5F6', '#C5E1A5']
        
        # Project to first 2 PCs (data is already in PCA space)
        t0_2d = timepoint_data['t0'].cpu().numpy()[:, :2]
        t7_2d = [timepoint_data[f't7_{i}'].cpu().numpy()[:, :2] for i in range(n_branches)]
        
        # Slice trajectories to first 2 PCs
        trajs_2d = []
        for traj in all_trajs:
            trajs_2d.append(traj.cpu().numpy()[:, :, :2])  # [n_samples, n_time, 2]
        
        # Compute axis limits from REAL CELLS ONLY
        all_coords_real = [t0_2d] + t7_2d
        all_coords_real = np.concatenate(all_coords_real, axis=0)
        x_min, x_max = all_coords_real[:, 0].min(), all_coords_real[:, 0].max()
        y_min, y_max = all_coords_real[:, 1].min(), all_coords_real[:, 1].max()
        x_margin = 0.05 * (x_max - x_min)
        y_margin = 0.05 * (y_max - y_min)
        x_min -= x_margin
        x_max += x_margin
        y_min -= y_margin
        y_max += y_margin
        
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Plot t=0 cells
        ax.scatter(t0_2d[:, 0], t0_2d[:, 1], 
                   c='#05009E', s=60, alpha=0.3, marker='x', 
                   label='t=0 cells', linewidth=1.5)
        
        # Plot each branch's cells and trajectories
        for i, traj_2d in enumerate(trajs_2d):
            c_end = branch_colors[i % len(branch_colors)]
            
            # Plot t=7 cells for this branch
            ax.scatter(t7_2d[i][:, 0], t7_2d[i][:, 1], 
                       c=c_end, s=60, alpha=0.3, marker='x', 
                       label=f't=7 (branch {i+1})', linewidth=1.5)
            
            # Plot continuous trajectories with LineCollection for speed
            cmap_colors = ["#05009E", "#A19EFF", c_end]
            cmap = LinearSegmentedColormap.from_list(f"veres_combined_cmap_{i}", cmap_colors)
            n_time = traj_2d.shape[1]
            segments = []
            seg_colors = []
            color_vals = cmap(np.linspace(0, 1, n_time))
            for j in range(traj_2d.shape[0]):
                pts = traj_2d[j]  # [T, 2]
                segs = np.stack([pts[:-1], pts[1:]], axis=1)  # [T-1, 2, 2]
                segments.append(segs)
                seg_colors.append(color_vals[:-1])
            segments = np.concatenate(segments, axis=0)
            seg_colors = np.concatenate(seg_colors, axis=0)
            lc = LineCollection(segments, colors=seg_colors, linewidths=1.5, alpha=0.6)
            ax.add_collection(lc)
            
            # Start and end points
            ax.scatter(traj_2d[:, 0, 0], traj_2d[:, 0, 1], 
                       c='#05009E', s=20, marker='o', 
                       zorder=5, edgecolors='white', linewidth=0.5, alpha=0.7)
            ax.scatter(traj_2d[:, -1, 0], traj_2d[:, -1, 1], 
                       c=c_end, s=20, marker='o', 
                       zorder=5, edgecolors='white', linewidth=0.5, alpha=0.7)
        
        ax.set_xlabel("PC 1", fontsize=14)
        ax.set_ylabel("PC 2", fontsize=14)
        ax.set_title(f"All {n_branches} Branch Trajectories (Veres) - PCA Projection", 
                     fontsize=16, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, frameon=False, ncol=2)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.args.data_name}_combined.png', dpi=300)
        plt.close()