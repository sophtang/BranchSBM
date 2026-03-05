import torch
import sys
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
import laspy
import numpy as np
from scipy.spatial import cKDTree
import math
from functools import partial
from torch.utils.data import TensorDataset


class GaussianMM:
    def __init__(self, mu, var):
        super().__init__()
        self.centers = torch.tensor(mu)
        self.logstd = torch.tensor(var).log() / 2.0
        self.K = self.centers.shape[0]

    def logprob(self, x):
        logprobs = self.normal_logprob(
            x.unsqueeze(1), self.centers.unsqueeze(0), self.logstd
        )
        logprobs = torch.sum(logprobs, dim=2)
        return torch.logsumexp(logprobs, dim=1) - math.log(self.K)

    def normal_logprob(self, z, mean, log_std):
        mean = mean + torch.tensor(0.0)
        log_std = log_std + torch.tensor(0.0)
        c = torch.tensor([math.log(2 * math.pi)]).to(z)
        inv_sigma = torch.exp(-log_std)
        tmp = (z - mean) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * log_std + c)

    def __call__(self, n_samples):
        idx = torch.randint(self.K, (n_samples,)).to(self.centers.device)
        mean = self.centers[idx]
        return torch.randn(*mean.shape).to(mean) * torch.exp(self.logstd) + mean
    
class BranchedLidarDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.max_dim = args.dim
        self.whiten = args.whiten
        self.p0_mu = [
            [-4.5, -4.0, 0.5],
            [-4.2, -3.5, 0.5],
            [-4.0, -3.0, 0.5],
            [-3.75, -2.5, 0.5],
        ]
        self.p0_var = 0.02
        
        self.p1_1_mu = [
            [-2.5, -0.25, 0.5],
            [-2.25, 0.675, 0.5],
            [-2, 1.5, 0.5],
        ]
        self.p1_2_mu = [
            [2, -2, 0.5], 
            [2.6, -1.25, 0.5], 
            [3.2, -0.5, 0.5]
        ]
        
        self.p1_var = 0.03
        self.k = 20
        self.n_samples = 5000
        self.num_timesteps = 2
        self.split_ratios = args.split_ratios
        self._prepare_data()
        
    def assign_region(self):
        all_centers = {
            0: torch.tensor(self.p0_mu),     # Region 0: p0
            1: torch.tensor(self.p1_1_mu),   # Region 1: p1_1
            2: torch.tensor(self.p1_2_mu),   # Region 2: p1_2
        }

        dataset = self.dataset.to(torch.float32)
        N = dataset.shape[0]
        assignments = torch.zeros(N, dtype=torch.long)

        # For each point, compute min distance to each region's centers
        for i in range(N):
            point = dataset[i]
            min_dist = float("inf")
            best_region = 0
            for region, centers in all_centers.items():
                dists = ((centers - point)**2).sum(dim=1)
                region_min = dists.min()
                if region_min < min_dist:
                    min_dist = region_min
                    best_region = region
            assignments[i] = best_region
        return assignments

    def _prepare_data(self):
        las = laspy.read(self.data_path)
        # Extract only "ground" points.
        self.mask = las.classification == 2
        # Original Preprocessing
        x_offset, x_scale = las.header.offsets[0], las.header.scales[0]
        y_offset, y_scale = las.header.offsets[1], las.header.scales[1]
        z_offset, z_scale = las.header.offsets[2], las.header.scales[2]
        dataset = np.vstack(
            (
                las.X[self.mask] * x_scale + x_offset,
                las.Y[self.mask] * y_scale + y_offset,
                las.Z[self.mask] * z_scale + z_offset,
            )
        ).transpose()
        mi = dataset.min(axis=0, keepdims=True)
        ma = dataset.max(axis=0, keepdims=True)
        dataset = (dataset - mi) / (ma - mi) * [10.0, 10.0, 2.0] + [-5.0, -5.0, 0.0]

        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        self.tree = cKDTree(dataset)

        x0_gaussian = GaussianMM(self.p0_mu, self.p0_var)(self.n_samples)
        x1_1_gaussian = GaussianMM(self.p1_1_mu, self.p1_var)(self.n_samples)
        x1_2_gaussian = GaussianMM(self.p1_2_mu, self.p1_var)(self.n_samples)

        x0 = self.get_tangent_proj(x0_gaussian)(x0_gaussian)
        x1_1 = self.get_tangent_proj(x1_1_gaussian)(x1_1_gaussian)
        x1_2 = self.get_tangent_proj(x1_2_gaussian)(x1_2_gaussian)

        split_index = int(self.n_samples * self.split_ratios[0])

        self.scaler = StandardScaler()
        if self.whiten:
            self.dataset = torch.tensor(
                self.scaler.fit_transform(dataset), dtype=torch.float32
            )
            x0 = torch.tensor(self.scaler.transform(x0), dtype=torch.float32)
            x1_1 = torch.tensor(self.scaler.transform(x1_1), dtype=torch.float32)
            x1_2 = torch.tensor(self.scaler.transform(x1_2), dtype=torch.float32)

        train_x0 = x0[:split_index]
        val_x0 = x0[split_index:]
        
        # branches
        train_x1_1 = x1_1[:split_index]
        print("train_x1_1")
        print(train_x1_1.shape)
        val_x1_1 = x1_1[split_index:]
        train_x1_2 = x1_2[:split_index]
        val_x1_2 = x1_2[split_index:]
        
        self.val_x0 = val_x0

        # Adjust split_index to ensure minimum validation samples
        if self.n_samples - split_index < self.batch_size:
            split_index = self.n_samples - self.batch_size

        self.train_dataloaders = {
            "x0": DataLoader(train_x0, batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_1": DataLoader(train_x1_1, batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_2": DataLoader(train_x1_2, batch_size=self.batch_size, shuffle=True, drop_last=True),
        }
        self.val_dataloaders = {
            "x0": DataLoader(val_x0, batch_size=self.batch_size, shuffle=False, drop_last=True),
            "x1_1": DataLoader(val_x1_1, batch_size=self.batch_size, shuffle=True, drop_last=True),
            "x1_2": DataLoader(val_x1_2, batch_size=self.batch_size, shuffle=True, drop_last=True),
        }
        # to edit?
        self.test_dataloaders = [
            DataLoader(
                self.val_x0,
                batch_size=self.val_x0.shape[0],
                shuffle=False,
                drop_last=False,
            ),
            DataLoader(
                self.dataset,
                batch_size=self.dataset.shape[0],
                shuffle=False,
                drop_last=False,
            ),
        ]
        
        points = self.dataset.cpu().numpy()
        x, y = points[:, 0], points[:, 1]
        # Diagonal-based coordinates (rotated 45°)
        u = (x + y) / np.sqrt(2)  # along x=y
        # start region (A) using u
        u_thresh = np.percentile(u, 30)  # tweak this threshold to control size
        mask_A = u <= u_thresh

        # among the rest, split by x=y diagonal
        remaining = ~mask_A
        mask_B = remaining & (x < y)  # left of diagonal
        mask_C = remaining & (x >= y)  # right of diagonal

        # Assign dataloaders
        self.metric_samples_dataloaders = [
            DataLoader(torch.tensor(points[mask_A], dtype=torch.float32), batch_size=points[mask_A].shape[0], shuffle=False),
            DataLoader(torch.tensor(points[mask_B], dtype=torch.float32), batch_size=points[mask_B].shape[0], shuffle=False),
            DataLoader(torch.tensor(points[mask_C], dtype=torch.float32), batch_size=points[mask_C].shape[0], shuffle=False),
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
        return CombinedLoader(self.test_dataloaders)

    def get_tangent_proj(self, points):
        w = self.get_tangent_plane(points)
        return partial(BranchedLidarDataModule.projection_op, w=w)

    def get_tangent_plane(self, points, temp=1e-3):
        points_np = points.detach().cpu().numpy()
        _, idx = self.tree.query(points_np, k=self.k)
        nearest_pts = self.dataset[idx]
        nearest_pts = torch.tensor(nearest_pts).to(points)

        dists = (points.unsqueeze(1) - nearest_pts).pow(2).sum(-1, keepdim=True)
        weights = torch.exp(-dists / temp)

        # Fits plane with least vertical distance.
        w = BranchedLidarDataModule.fit_plane(nearest_pts, weights)
        return w

    @staticmethod
    def fit_plane(points, weights=None):
        """Expects points to be of shape (..., 3).
        Returns [a, b, c] such that the plane is defined as
            ax + by + c = z
        """
        D = torch.cat([points[..., :2], torch.ones_like(points[..., 2:3])], dim=-1)
        z = points[..., 2]
        if weights is not None:
            Dtrans = D.transpose(-1, -2)
        else:
            DW = D * weights
            Dtrans = DW.transpose(-1, -2)
        w = torch.linalg.solve(
            torch.matmul(Dtrans, D), torch.matmul(Dtrans, z.unsqueeze(-1))
        ).squeeze(-1)
        return w

    @staticmethod
    def projection_op(x, w):
        """Projects points to a plane defined by w."""
        # Normal vector to the tangent plane.
        n = torch.cat([w[..., :2], -torch.ones_like(w[..., 2:3])], dim=1)

        pn = torch.sum(x * n, dim=-1, keepdim=True)
        nn = torch.sum(n * n, dim=-1, keepdim=True)

        # Offset.
        d = w[..., 2:3]

        # Projection of x onto n.
        projn_x = ((pn + d) / nn) * n

        # Remove component in the normal direction.
        return x - projn_x

class WeightedBranchedLidarDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()

        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.max_dim = args.dim
        self.whiten = args.whiten
        self.p0_mu = [
            [-4.5, -4.0, 0.5],
            [-4.2, -3.5, 0.5],
            [-4.0, -3.0, 0.5],
            [-3.75, -2.5, 0.5],
        ]
        self.p0_var = 0.02
        # multiple p1 for each branch
        #changed
        self.p1_1_mu = [
            [-2.5, -0.25, 0.5],
            [-2.25, 0.675, 0.5],
            [-2, 1.5, 0.5],
        ]
        self.p1_2_mu = [
            [2, -2, 0.5], 
            [2.6, -1.25, 0.5], 
            [3.2, -0.5, 0.5]
        ]
        
        self.p1_var = 0.03
        self.k = 20
        self.n_samples = 5000
        self.num_timesteps = 2
        self.split_ratios = args.split_ratios
        
        self.num_timesteps = 2
        self.metric_clusters = 3
        self.args = args
        self._prepare_data()

    def _prepare_data(self):
        las = laspy.read(self.data_path)
        # Extract only "ground" points.
        self.mask = las.classification == 2
        # Original Preprocessing
        x_offset, x_scale = las.header.offsets[0], las.header.scales[0]
        y_offset, y_scale = las.header.offsets[1], las.header.scales[1]
        z_offset, z_scale = las.header.offsets[2], las.header.scales[2]
        dataset = np.vstack(
            (
                las.X[self.mask] * x_scale + x_offset,
                las.Y[self.mask] * y_scale + y_offset,
                las.Z[self.mask] * z_scale + z_offset,
            )
        ).transpose()
        mi = dataset.min(axis=0, keepdims=True)
        ma = dataset.max(axis=0, keepdims=True)
        dataset = (dataset - mi) / (ma - mi) * [10.0, 10.0, 2.0] + [-5.0, -5.0, 0.0]

        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        self.tree = cKDTree(dataset)

        x0_gaussian = GaussianMM(self.p0_mu, self.p0_var)(self.n_samples)
        x1_1_gaussian = GaussianMM(self.p1_1_mu, self.p1_var)(self.n_samples)
        x1_2_gaussian = GaussianMM(self.p1_2_mu, self.p1_var)(self.n_samples)

        x0 = self.get_tangent_proj(x0_gaussian)(x0_gaussian)
        x1_1 = self.get_tangent_proj(x1_1_gaussian)(x1_1_gaussian)
        x1_2 = self.get_tangent_proj(x1_2_gaussian)(x1_2_gaussian)

        split_index = int(self.n_samples * self.split_ratios[0])

        self.scaler = StandardScaler()
        if self.whiten:
            self.dataset = torch.tensor(
                self.scaler.fit_transform(dataset), dtype=torch.float32
            )
            x0 = torch.tensor(self.scaler.transform(x0), dtype=torch.float32)
            x1_1 = torch.tensor(self.scaler.transform(x1_1), dtype=torch.float32)
            x1_2 = torch.tensor(self.scaler.transform(x1_2), dtype=torch.float32)

        self.coords_t0 = x0
        self.coords_t1_1 = x1_1
        self.coords_t1_2 = x1_2
        self.time_labels = np.concatenate([
            np.zeros(len(self.coords_t0)),    # t=0
            np.ones(len(self.coords_t1_1)),     # t=1
            np.ones(len(self.coords_t1_2)),     # t=1
        ])
        
        train_x0 = x0[:split_index]
        val_x0 = x0[split_index:]
        
        # branches
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

        # Adjust split_index to ensure minimum validation samples
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
        
        # to edit?
        self.test_dataloaders = {
            "x0": DataLoader(TensorDataset(val_x0, val_x0_weights), batch_size=self.val_x0.shape[0], shuffle=False, drop_last=False),
            "x1_1": DataLoader(TensorDataset(val_x1_1, val_x1_1_weights), batch_size=self.val_x0.shape[0], shuffle=True, drop_last=True),
            "x1_2": DataLoader(TensorDataset(val_x1_2, val_x1_2_weights), batch_size=self.val_x0.shape[0], shuffle=True, drop_last=True),
            "dataset": DataLoader(TensorDataset(self.dataset), batch_size=self.dataset.shape[0], shuffle=False, drop_last=False),
        }
        
        points = self.dataset.cpu().numpy()
        x, y = points[:, 0], points[:, 1]
        # Diagonal-based coordinates (rotated 45°)
        u = (x + y) / np.sqrt(2)  # along x=y
        # start region (A) using u
        u_thresh = np.percentile(u, 30)  # tweak this threshold to control size
        mask_A = u <= u_thresh

        # among the rest, split by x=y diagonal
        remaining = ~mask_A
        mask_B = remaining & (x < y)  # left of diagonal
        mask_C = remaining & (x >= y)  # right of diagonal

        # Assign dataloaders
        self.metric_samples_dataloaders = [
            DataLoader(torch.tensor(points[mask_A], dtype=torch.float32), batch_size=points[mask_A].shape[0], shuffle=False),
            DataLoader(torch.tensor(points[mask_B], dtype=torch.float32), batch_size=points[mask_B].shape[0], shuffle=False),
            DataLoader(torch.tensor(points[mask_C], dtype=torch.float32), batch_size=points[mask_C].shape[0], shuffle=False),
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

    def get_tangent_proj(self, points):
        w = self.get_tangent_plane(points)
        return partial(BranchedLidarDataModule.projection_op, w=w)

    def get_tangent_plane(self, points, temp=1e-3):
        points_np = points.detach().cpu().numpy()
        _, idx = self.tree.query(points_np, k=self.k)
        nearest_pts = self.dataset[idx]
        nearest_pts = torch.tensor(nearest_pts).to(points)

        dists = (points.unsqueeze(1) - nearest_pts).pow(2).sum(-1, keepdim=True)
        weights = torch.exp(-dists / temp)

        # Fits plane with least vertical distance.
        w = BranchedLidarDataModule.fit_plane(nearest_pts, weights)
        return w

    @staticmethod
    def fit_plane(points, weights=None):
        """Expects points to be of shape (..., 3).
        Returns [a, b, c] such that the plane is defined as
            ax + by + c = z
        """
        D = torch.cat([points[..., :2], torch.ones_like(points[..., 2:3])], dim=-1)
        z = points[..., 2]
        if weights is not None:
            Dtrans = D.transpose(-1, -2)
        else:
            DW = D * weights
            Dtrans = DW.transpose(-1, -2)
        w = torch.linalg.solve(
            torch.matmul(Dtrans, D), torch.matmul(Dtrans, z.unsqueeze(-1))
        ).squeeze(-1)
        return w

    @staticmethod
    def projection_op(x, w):
        """Projects points to a plane defined by w."""
        # Normal vector to the tangent plane.
        n = torch.cat([w[..., :2], -torch.ones_like(w[..., 2:3])], dim=1)

        pn = torch.sum(x * n, dim=-1, keepdim=True)
        nn = torch.sum(n * n, dim=-1, keepdim=True)

        # Offset.
        d = w[..., 2:3]

        # Projection of x onto n.
        projn_x = ((pn + d) / nn) * n

        # Remove component in the normal direction.
        return x - projn_x
    
    def get_timepoint_data(self):
        """Return data organized by timepoints for visualization"""
        return {
            't0': self.coords_t0,
            't1_1': self.coords_t1_1, 
            't1_2': self.coords_t1_2, 
            'time_labels': self.time_labels
        }

def get_datamodule():
    datamodule = WeightedBranchedLidarDataModule(args)
    datamodule.setup(stage="fit")   
    return datamodule