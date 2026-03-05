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
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset

class WeightedBranchedCellDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.max_dim = args.dim
        self.whiten = args.whiten
        self.k = 20
        self.n_samples = 1429 
        self.num_timesteps = 2  # t=0, t=1, t=2
        self.split_ratios = args.split_ratios
        self.metric_clusters = args.metric_clusters
        self.args = args
        self._prepare_data()
        

    def _prepare_data(self):
        print("Preparing cell data in BranchedCellDataModule")
        
        df = pd.read_csv(self.data_path)
        
        # Build dictionary of coordinates by time
        coords_by_t = {
            t: df[df["samples"] == t][["x1","x2"]].values
            for t in sorted(df["samples"].unique())
        }
        n0 = coords_by_t[0].shape[0]  # Number of T=0 points
        self.n_samples = n0  # Update n_samples to match actual data if changes

        # Cluster the t=2 cells into two branches
        km = KMeans(n_clusters=2, random_state=42).fit(coords_by_t[2])
        df2 = df[df["samples"] == 2].copy()
        df2["branch"] = km.labels_
        
        cluster_counts = df2["branch"].value_counts().sort_index()
        print(cluster_counts)

        # Sample n0 points from each branch
        endpoints = {}
        for b in (0, 1):
            endpoints[b] = (
                df2[df2["branch"] == b]
                .sample(n=n0, random_state=42)[["x1","x2"]]
                .values
            )
            
        x0 = torch.tensor(coords_by_t[0], dtype=torch.float32) # T=0 coordinates index
        x_inter = torch.tensor(coords_by_t[1], dtype=torch.float32)
        x1_1 = torch.tensor(endpoints[0], dtype=torch.float32) # Branch index
        x1_2 = torch.tensor(endpoints[1], dtype=torch.float32) # Branch index

        self.coords_t0 = x0
        self.coords_t1 = x_inter
        self.coords_t2_1 = x1_1
        self.coords_t2_2 = x1_2
        self.time_labels = np.concatenate([
            np.zeros(len(self.coords_t0)),    # t=0
            np.ones(len(self.coords_t1)),     # t=1
            np.ones(len(self.coords_t2_1)) * 2,     # t=1
            np.ones(len(self.coords_t2_2)) * 2,
        ])
        
        split_index = int(n0 * self.split_ratios[0])
        
        if n0 - split_index < self.batch_size:
            split_index = n0 - self.batch_size

        train_x0 = x0[:split_index]
        val_x0 = x0[split_index:]
        train_x1_1 = x1_1[:split_index]
        val_x1_1 = x1_1[split_index:]
        train_x1_2 = x1_2[:split_index]
        val_x1_2 = x1_2[split_index:]
        
        self.val_x0 = val_x0
        
        train_x0_weights = torch.full((train_x0.shape[0], 1), fill_value=1.0)
        train_x1_1_weights = torch.full((train_x1_1.shape[0], 1), fill_value=0.5)
        train_x1_2_weights = torch.full((train_x1_2.shape[0], 1), fill_value=0.5)
        
        val_x0_weights = torch.full((val_x0.shape[0], 1), fill_value=1.0)
        val_x1_1_weights = torch.full((val_x1_1.shape[0], 1), fill_value=0.5)
        val_x1_2_weights = torch.full((val_x1_2.shape[0], 1), fill_value=0.5)

        if self.n_samples - split_index < self.batch_size:
            split_index = self.n_samples - self.batch_size
            
        self.train_dataloaders = {
            "x0": DataLoader(TensorDataset(train_x0, train_x0_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_1": DataLoader(TensorDataset(train_x1_1, train_x1_1_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_2": DataLoader(TensorDataset(train_x1_2, train_x1_2_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
        }
        
        self.val_dataloaders = {
            "x0": DataLoader(TensorDataset(val_x0, val_x0_weights), batch_size=self.batch_size, shuffle=False, drop_last=True),
            "x1_1": DataLoader(TensorDataset(val_x1_1, val_x1_1_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_2": DataLoader(TensorDataset(val_x1_2, val_x1_2_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
        }

        all_data = np.vstack([coords_by_t[t] for t in sorted(coords_by_t.keys())])
        self.dataset = torch.tensor(all_data, dtype=torch.float32)
        self.tree = cKDTree(all_data)
        
        # if whitening is enabled, need to apply this to the full dataset
        #if self.whiten:
            #self.scaler = StandardScaler()
            #self.dataset = torch.tensor(
                #self.scaler.fit_transform(all_data), dtype=torch.float32
            #)
            
        self.test_dataloaders = {
            "x0": DataLoader(TensorDataset(val_x0, val_x0_weights), batch_size=self.val_x0.shape[0], shuffle=False, drop_last=False),
            "dataset": DataLoader(TensorDataset(self.dataset), batch_size=self.dataset.shape[0], shuffle=False, drop_last=False),
        }
        
        # Metric Dataloader
        # K-means clustering of ALL points into 2 groups
        if self.metric_clusters == 3:
            km_all = KMeans(n_clusters=3, random_state=45).fit(self.dataset.numpy())
            cluster_labels = km_all.labels_
            
            cluster_0_mask = cluster_labels == 0
            cluster_1_mask = cluster_labels == 1
            cluster_2_mask = cluster_labels == 2
            
            samples = self.dataset.cpu().numpy()
            
            cluster_0_data = samples[cluster_0_mask]
            cluster_1_data = samples[cluster_1_mask]
            cluster_2_data = samples[cluster_2_mask]
            
            self.metric_samples_dataloaders = [
                DataLoader(
                    torch.tensor(cluster_1_data, dtype=torch.float32),
                    batch_size=cluster_1_data.shape[0],
                    shuffle=False,
                    drop_last=False,
                ),
                DataLoader(
                    torch.tensor(cluster_2_data, dtype=torch.float32),
                    batch_size=cluster_2_data.shape[0],
                    shuffle=False,
                    drop_last=False,
                ),
                
                DataLoader(
                    torch.tensor(cluster_0_data, dtype=torch.float32),
                    batch_size=cluster_0_data.shape[0],
                    shuffle=False,
                    drop_last=False,
                ),
            ]
        else:
            km_all = KMeans(n_clusters=2, random_state=45).fit(self.dataset.numpy())
            cluster_labels = km_all.labels_
            
            cluster_0_mask = cluster_labels == 0
            cluster_1_mask = cluster_labels == 1
            
            samples = self.dataset.cpu().numpy()
            
            cluster_0_data = samples[cluster_0_mask]
            cluster_1_data = samples[cluster_1_mask]
            
            self.metric_samples_dataloaders = [
                DataLoader(
                    torch.tensor(cluster_1_data, dtype=torch.float32),
                    batch_size=cluster_1_data.shape[0],
                    shuffle=False,
                    drop_last=False,
                ),
                DataLoader(
                    torch.tensor(cluster_0_data, dtype=torch.float32),
                    batch_size=cluster_0_data.shape[0],
                    shuffle=False,
                    drop_last=False,
                ),
            ]


    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        combined_loaders = {
            "test_samples": CombinedLoader(self.test_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def get_manifold_proj(self, points):
        """Adapted for 2D cell data - uses local neighborhood averaging instead of plane fitting"""
        return partial(self.local_smoothing_op, tree=self.tree, dataset=self.dataset)

    @staticmethod
    def local_smoothing_op(x, tree, dataset, k=10, temp=1e-3):
        """
        Apply local smoothing based on k-nearest neighbors in the full dataset
        This replaces the plane projection for 2D manifold regularization
        """
        points_np = x.detach().cpu().numpy()
        _, idx = tree.query(points_np, k=k)
        nearest_pts = dataset[idx]  # Shape: (batch_size, k, 2)
        
        # Compute weighted average of neighbors
        dists = (x.unsqueeze(1) - nearest_pts).pow(2).sum(-1, keepdim=True)
        weights = torch.exp(-dists / temp)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Weighted average of neighbors
        smoothed = (weights * nearest_pts).sum(dim=1)
        
        # Blend original point with smoothed version
        alpha = 0.3  # How much smoothing to apply
        return (1 - alpha) * x + alpha * smoothed
    
    def get_timepoint_data(self):
        """Return data organized by timepoints for visualization"""
        return {
            't0': self.coords_t0,
            't1': self.coords_t1, 
            't2_1': self.coords_t2_1, 
            't2_2': self.coords_t2_2, 
            'time_labels': self.time_labels
        }



class SingleBranchCellDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.max_dim = args.dim
        self.whiten = args.whiten
        self.k = 20
        self.n_samples = 1429 
        self.num_timesteps = 3  # t=0, t=1, t=2
        self.split_ratios = args.split_ratios
        self.metric_clusters = 3
        self.args = args
        self._prepare_data()
        

    def _prepare_data(self):
        print("Preparing cell data in BranchedCellDataModule")
        
        df = pd.read_csv(self.data_path)
        
        # Build dictionary of coordinates by time
        coords_by_t = {
            t: df[df["samples"] == t][["x1","x2"]].values
            for t in sorted(df["samples"].unique())
        }
        n0 = coords_by_t[0].shape[0]  # Number of T=0 points
        self.n_samples = n0  # Update n_samples to match actual data if changes

        x0 = torch.tensor(coords_by_t[0], dtype=torch.float32) # T=0 coordinates index
        x_inter = torch.tensor(coords_by_t[1], dtype=torch.float32)
        x1 = torch.tensor(coords_by_t[2], dtype=torch.float32) # Branch index

        # Store for get_timepoint_data()
        self.coords_t0 = x0
        self.coords_t1 = x_inter
        self.coords_t2 = x1
        self.time_labels = np.concatenate([
            np.zeros(len(x0)),
            np.ones(len(x_inter)),
            np.ones(len(x1)) * 2,
        ])

        split_index = int(n0 * self.split_ratios[0])
        
        if n0 - split_index < self.batch_size:
            split_index = n0 - self.batch_size

        train_x0 = x0[:split_index]
        val_x0 = x0[split_index:]
        train_x1 = x1[:split_index]
        val_x1 = x1[split_index:]
        
        self.val_x0 = val_x0
        
        train_x0_weights = torch.full((train_x0.shape[0], 1), fill_value=1.0)
        train_x1_weights = torch.full((train_x1.shape[0], 1), fill_value=0.5)
        
        val_x0_weights = torch.full((val_x0.shape[0], 1), fill_value=1.0)
        val_x1_weights = torch.full((val_x1.shape[0], 1), fill_value=0.5)

        if self.n_samples - split_index < self.batch_size:
            split_index = self.n_samples - self.batch_size
            
        self.train_dataloaders = {
            "x0": DataLoader(TensorDataset(train_x0, train_x0_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1": DataLoader(TensorDataset(train_x1, train_x1_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
        }
        
        self.val_dataloaders = {
            "x0": DataLoader(TensorDataset(val_x0, val_x0_weights), batch_size=self.batch_size, shuffle=False, drop_last=True),
            "x1": DataLoader(TensorDataset(val_x1, val_x1_weights), batch_size=self.batch_size, shuffle=True, drop_last=True),
        }

        all_data = np.vstack([coords_by_t[t] for t in sorted(coords_by_t.keys())])
        self.dataset = torch.tensor(all_data, dtype=torch.float32)
        self.tree = cKDTree(all_data)
        
        # if whitening is enabled, need to apply this to the full dataset
        if self.whiten:
            self.scaler = StandardScaler()
            self.dataset = torch.tensor(
                self.scaler.fit_transform(all_data), dtype=torch.float32
            )
            
        self.test_dataloaders = {
            "x0": DataLoader(TensorDataset(val_x0, val_x0_weights), batch_size=self.val_x0.shape[0], shuffle=False, drop_last=False),
            "dataset": DataLoader(TensorDataset(self.dataset), batch_size=self.dataset.shape[0], shuffle=False, drop_last=False),
        }
        
        # Metric Dataloader
        # K-means clustering of ALL points into 2 groups
        km_all = KMeans(n_clusters=2, random_state=45).fit(self.dataset.numpy())
        cluster_labels = km_all.labels_
        
        cluster_0_mask = cluster_labels == 0
        cluster_1_mask = cluster_labels == 1
        
        samples = self.dataset.cpu().numpy()
        
        cluster_0_data = samples[cluster_0_mask]
        cluster_1_data = samples[cluster_1_mask]
        
        self.metric_samples_dataloaders = [
            DataLoader(
                torch.tensor(cluster_1_data, dtype=torch.float32),
                batch_size=cluster_1_data.shape[0],
                shuffle=False,
                drop_last=False,
            ),
            DataLoader(
                torch.tensor(cluster_0_data, dtype=torch.float32),
                batch_size=cluster_0_data.shape[0],
                shuffle=False,
                drop_last=False,
            ),
        ]


    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        combined_loaders = {
            "test_samples": CombinedLoader(self.test_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def get_manifold_proj(self, points):
        """Adapted for 2D cell data - uses local neighborhood averaging instead of plane fitting"""
        return partial(self.local_smoothing_op, tree=self.tree, dataset=self.dataset)

    @staticmethod
    def local_smoothing_op(x, tree, dataset, k=10, temp=1e-3):
        """
        Apply local smoothing based on k-nearest neighbors in the full dataset
        This replaces the plane projection for 2D manifold regularization
        """
        points_np = x.detach().cpu().numpy()
        _, idx = tree.query(points_np, k=k)
        nearest_pts = dataset[idx]  # Shape: (batch_size, k, 2)
        
        # Compute weighted average of neighbors
        dists = (x.unsqueeze(1) - nearest_pts).pow(2).sum(-1, keepdim=True)
        weights = torch.exp(-dists / temp)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Weighted average of neighbors
        smoothed = (weights * nearest_pts).sum(dim=1)
        
        # Blend original point with smoothed version
        alpha = 0.3  # How much smoothing to apply
        return (1 - alpha) * x + alpha * smoothed

    def get_timepoint_data(self):
        """Return data organized by timepoints for visualization"""
        return {
            't0': self.coords_t0,
            't1': self.coords_t1,
            't2': self.coords_t2,
            'time_labels': self.time_labels
        }

"""def get_datamodule():
    datamodule = WeightedBranchedCellDataModule(args)
    datamodule.setup(stage="fit")   
    return datamodule"""