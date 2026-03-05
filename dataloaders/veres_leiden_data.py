import torch
import sys
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import numpy as np
from scipy.spatial import cKDTree
import math
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.neighbors import kneighbors_graph
import igraph as ig
from leidenalg import find_partition, ModularityVertexPartition

class WeightedBranchedVeresDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.max_dim = args.dim
        self.whiten = args.whiten
        self.k = 20
        self.num_timesteps = 8
        # initial placeholder, will be set by clustering result
        self.num_branches = args.branches if hasattr(args, 'branches') else None
        self.split_ratios = args.split_ratios
        self.metric_clusters = args.metric_clusters
        self.discard_small = args.discard if hasattr(args, 'discard') else False
        self.args = args
        self._prepare_data()

    def _prepare_data(self):
        print("Preparing Veres cell data with Leiden clustering in WeightedBranchedVeresLeidenDataModule")
        df = pd.read_csv(self.data_path)

        # Build dictionary of coordinates by time
        coords_by_t = {
            t: df[df["samples"] == t].iloc[:, 1:].values  # Skip 'samples' column
            for t in sorted(df["samples"].unique())
        }

        n0 = coords_by_t[0].shape[0]
        self.n_samples = n0

        print("Timepoint distribution:")
        for t in sorted(coords_by_t.keys()):
            print(f"  t={t}: {coords_by_t[t].shape[0]} points")

        # Leiden clustering on final timepoint
        final_t = max(coords_by_t.keys())
        coords_final = coords_by_t[final_t]
        k = 20
        knn_graph = kneighbors_graph(coords_final, k, mode='connectivity', include_self=False)
        sources, targets = knn_graph.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        graph = ig.Graph(edgelist, directed=False)
        partition = find_partition(graph, ModularityVertexPartition)
        leiden_labels = np.array(partition.membership)
        n_leiden = len(np.unique(leiden_labels))
        print(f"Leiden found {n_leiden} clusters at t={final_t}")

        df_final = df[df["samples"] == final_t].copy()
        df_final["branch"] = leiden_labels

        cluster_counts = df_final["branch"].value_counts().sort_index()
        print(f"Branch distribution at t={final_t} (pre-merge):")
        print(cluster_counts)

        # Merge small clusters to nearest large cluster (by centroid)
        min_cells = 100  # threshold; adjust if needed
        cluster_data_dict = {}
        cluster_sizes = []
        for b in range(n_leiden):
            branch_data = df_final[df_final["branch"] == b].iloc[:, 1:-1].values
            cluster_data_dict[b] = branch_data
            cluster_sizes.append(branch_data.shape[0])

        large_clusters = [b for b, size in enumerate(cluster_sizes) if size >= min_cells]
        small_clusters = [b for b, size in enumerate(cluster_sizes) if size < min_cells]

        # If no large cluster exists (all small), treat all clusters as large
        if len(large_clusters) == 0:
            large_clusters = list(range(n_leiden))
            small_clusters = []

        if self.discard_small:
            # Discard small clusters instead of merging
            print(f"Discarding {len(small_clusters)} small clusters (< {min_cells} cells)")
            # Keep only cells from large clusters
            mask = np.isin(leiden_labels, large_clusters)
            df_final = df_final[mask].copy()
            merged_labels = leiden_labels[mask]
            
            # Remap to contiguous ids
            new_ids = np.unique(merged_labels)
            id_map = {old: new for new, old in enumerate(new_ids)}
            merged_labels = np.array([id_map[x] for x in merged_labels])
            n_merged = len(np.unique(merged_labels))
            
            df_final["branch"] = merged_labels
            print(f"Kept {n_merged} large clusters")
        else:
            centroids = {b: np.mean(cluster_data_dict[b], axis=0) for b in range(n_leiden) if cluster_data_dict[b].shape[0] > 0}

            merged_labels = leiden_labels.copy()
            for b in small_clusters:
                if cluster_data_dict[b].shape[0] == 0:
                    continue
                # find nearest large cluster
                dists = [np.linalg.norm(centroids[b] - centroids[bl]) for bl in large_clusters]
                nearest_large = large_clusters[int(np.argmin(dists))]
                merged_labels[leiden_labels == b] = nearest_large

            # remap to contiguous ids
            new_ids = np.unique(merged_labels)
            id_map = {old: new for new, old in enumerate(new_ids)}
            merged_labels = np.array([id_map[x] for x in merged_labels])
            n_merged = len(np.unique(merged_labels))

            df_final["branch"] = merged_labels
            print(f"Merged into {n_merged} clusters")
            
        cluster_counts_merged = df_final["branch"].value_counts().sort_index()
        print(f"Branch distribution at t={final_t} (post-merge):")
        print(cluster_counts_merged)

        endpoints = {}
        cluster_sizes = []
        for b in range(n_merged):
            branch_data = df_final[df_final["branch"] == b].iloc[:, 1:-1].values
            cluster_sizes.append(branch_data.shape[0])
            replace = branch_data.shape[0] < n0
            sampled_indices = np.random.choice(branch_data.shape[0], size=n0, replace=replace)
            endpoints[b] = branch_data[sampled_indices]
        total_t_final = sum(cluster_sizes)

        x0 = torch.tensor(coords_by_t[0], dtype=torch.float32)
        self.coords_t0 = x0
        # intermediate timepoints
        self.coords_intermediate = {t: torch.tensor(coords_by_t[t], dtype=torch.float32)
                                    for t in coords_by_t.keys() if t != 0 and t != final_t}

        self.branch_endpoints = {b: torch.tensor(endpoints[b], dtype=torch.float32) for b in range(n_merged)}
        self.num_branches = n_merged

        # time labels (for visualization)
        time_labels_list = [np.zeros(len(self.coords_t0))]
        for t in sorted(self.coords_intermediate.keys()):
            time_labels_list.append(np.ones(len(self.coords_intermediate[t])) * t)
        for b in range(self.num_branches):
            time_labels_list.append(np.ones(len(self.branch_endpoints[b])) * final_t)
        self.time_labels = np.concatenate(time_labels_list)

        # splits
        split_index = int(n0 * self.split_ratios[0])
        if n0 - split_index < self.batch_size:
            split_index = n0 - self.batch_size

        train_x0 = x0[:split_index]
        val_x0 = x0[split_index:]
        self.val_x0 = val_x0

        train_x0_weights = torch.full((train_x0.shape[0], 1), fill_value=1.0)
        val_x0_weights = torch.full((val_x0.shape[0], 1), fill_value=1.0)

        # branch weights proportional to cluster sizes
        branch_weights = [size / total_t_final for size in cluster_sizes]

        # Split intermediate timepoints for sequential training support
        train_intermediate = {}
        val_intermediate = {}
        self.train_coords_intermediate = {}  # Store training-only intermediate data for MMD
        for t in sorted(self.coords_intermediate.keys()):
            coords_t = self.coords_intermediate[t]
            train_coords_t = coords_t[:split_index]
            val_coords_t = coords_t[split_index:]
            train_weights_t = torch.full((train_coords_t.shape[0], 1), fill_value=1.0)
            val_weights_t = torch.full((val_coords_t.shape[0], 1), fill_value=1.0)
            train_intermediate[f"x{t}"] = (train_coords_t, train_weights_t)
            val_intermediate[f"x{t}"] = (val_coords_t, val_weights_t)
            self.train_coords_intermediate[t] = train_coords_t  # Store training data by int key

        train_loaders = {
            "x0": DataLoader(TensorDataset(train_x0, train_x0_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
        }
        val_loaders = {
            "x0": DataLoader(TensorDataset(val_x0, val_x0_weights), batch_size=self.batch_size, shuffle=False, drop_last=True),
        }

        # Add all intermediate timepoints to loaders
        for t_key in sorted(train_intermediate.keys()):
            train_coords_t, train_weights_t = train_intermediate[t_key]
            val_coords_t, val_weights_t = val_intermediate[t_key]
            train_loaders[t_key] = DataLoader(
                TensorDataset(train_coords_t, train_weights_t),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )
            val_loaders[t_key] = DataLoader(
                TensorDataset(val_coords_t, val_weights_t),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True
            )

        for b in range(self.num_branches):
            # Calculate split based on this branch's size, not t=0 size
            branch_size = self.branch_endpoints[b].shape[0]
            branch_split_index = int(branch_size * self.split_ratios[0])
            if branch_size - branch_split_index < self.batch_size:
                branch_split_index = max(0, branch_size - self.batch_size)
            
            train_branch = self.branch_endpoints[b][:branch_split_index]
            val_branch = self.branch_endpoints[b][branch_split_index:]
            train_branch_weights = torch.full((train_branch.shape[0], 1), fill_value=branch_weights[b])
            val_branch_weights = torch.full((val_branch.shape[0], 1), fill_value=branch_weights[b])
            train_loaders[f"x1_{b+1}"] = DataLoader(
                TensorDataset(train_branch, train_branch_weights),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )
            val_loaders[f"x1_{b+1}"] = DataLoader(
                TensorDataset(val_branch, val_branch_weights),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )

        self.train_dataloaders = train_loaders
        self.val_dataloaders = val_loaders

        # full dataset
        all_data_list = [coords_by_t[t] for t in sorted(coords_by_t.keys())]
        all_data = np.vstack(all_data_list)
        self.dataset = torch.tensor(all_data, dtype=torch.float32)
        self.tree = cKDTree(all_data)

        self.test_dataloaders = {
            "x0": DataLoader(TensorDataset(self.val_x0, val_x0_weights), batch_size=self.val_x0.shape[0], shuffle=False, drop_last=False),
            "dataset": DataLoader(TensorDataset(self.dataset), batch_size=self.dataset.shape[0], shuffle=False, drop_last=False),
        }

        # Metric dataloaders: t0 vs (t1..t_final + endpoints)
        cluster_0_data = self.coords_t0.cpu().numpy()
        cluster_1_list = [self.coords_intermediate[t].cpu().numpy() for t in sorted(self.coords_intermediate.keys())]
        cluster_1_list.extend([self.branch_endpoints[b].cpu().numpy() for b in range(self.num_branches)])
        cluster_1_data = np.vstack(cluster_1_list)

        self.metric_samples_dataloaders = [
            DataLoader(torch.tensor(cluster_0_data, dtype=torch.float32), batch_size=cluster_0_data.shape[0], shuffle=False, drop_last=False),
            DataLoader(torch.tensor(cluster_1_data, dtype=torch.float32), batch_size=cluster_1_data.shape[0], shuffle=False, drop_last=False),
        ]

    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(self.metric_samples_dataloaders, mode="min_size"),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(self.metric_samples_dataloaders, mode="min_size"),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        combined_loaders = {
            "test_samples": CombinedLoader(self.test_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(self.metric_samples_dataloaders, mode="min_size"),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def get_manifold_proj(self, points):
        return partial(self.local_smoothing_op, tree=self.tree, dataset=self.dataset)

    @staticmethod
    def local_smoothing_op(x, tree, dataset, k=10, temp=1e-3):
        points_np = x.detach().cpu().numpy()
        _, idx = tree.query(points_np, k=k)
        nearest_pts = dataset[idx]
        dists = (x.unsqueeze(1) - nearest_pts).pow(2).sum(-1, keepdim=True)
        weights = torch.exp(-dists / temp)
        weights = weights / weights.sum(dim=1, keepdim=True)
        smoothed = (weights * nearest_pts).sum(dim=1)
        alpha = 0.3
        return (1 - alpha) * x + alpha * smoothed

    def get_timepoint_data(self):
        result = {
            't0': self.coords_t0,
            'time_labels': self.time_labels
        }
        # intermediate timepoints
        for t in sorted(self.coords_intermediate.keys()):
            result[f't{t}'] = self.coords_intermediate[t]
        final_t = max([0] + list(self.coords_intermediate.keys())) + 1
        for b in range(self.num_branches):
            result[f't{final_t}_{b}'] = self.branch_endpoints[b]
        return result

    def get_train_intermediate_data(self):
        if hasattr(self, 'train_coords_intermediate'):
            return self.train_coords_intermediate
        else:
            # Fallback to full intermediate data if train split not available
            print("Warning: train_coords_intermediate not found, returning full intermediate data.")
            return self.coords_intermediate
