# Running Experiments with BranchSBM 🌳🧫

This directory contains training scripts for all experiments with BranchSBM, including LiDAR navigation 🗻, simulating cell differentiation 🧫, and cell state perturbation modelling 🧫. This codebase contains code from the [Metric Flow Matching repo](https://github.com/kkapusniak/metric-flow-matching) ([Kapusniak et al. 2024](https://arxiv.org/abs/2405.14780)).

## Environment Installation
```
conda env create -f environment.yml

conda activate branchsbm
```

## Data
LiDAR data is taken from the [Generalized Schrödinger Bridge Matching repo](https://github.com/facebookresearch/generalized-schrodinger-bridge-matching) and Mouse Hematopoesis is taken from the [DeepRUOT repo](https://github.com/zhenyiizhang/DeepRUOT)

We use perturbation data from the [Tahoe-100M dataset](https://huggingface.co/datasets/tahoebio/Tahoe-100M) containing control DMSO-treated cell data and perturbed cell data. 

The raw data contains a total of 60K genes. We select the top 2000 highly variable genes (HVGs) and perform principal component analysis (PCA), to maximally capture the variance in the data via the top principal components (38% in the top-50 PCs). **Our goal is to learn the dynamic trajectories that map control cell clusters to the perturbed cell clusters.**

**Specifically, we model the following perturbations**:

1. **Clonidine**: Cell states under 5uM Clonidine perturbation at various PC dimensions (50D, 100D, 150D) with 1 unseen population.
2. **Trametinib**: Cell states under 5uM Trametinib perturbation (50D) with 2 unseen populations.

All data files are stored in:
```
BranchSBMl/data/
├── rainier2-thin.las # LiDAR data
├── mouse_hematopoiesis.csv # Mouse Hematopoiesis data
├── pca_and_leiden_labels.csv # Clonidine data
├── Trametinib_5.0uM_pca_and_leidenumap_labels.csv # Trametinib data
└── Veres_alltime.csv # Pancreatic β-Cell data
```

## Running Experiments

All training scripts are located in `BranchSBM/scripts/`. Each script is pre-configured for a specific experiment.

The scripts for BranchSBM experiments include:

- **`lidar.sh`** - LiDAR trajectory data with 2 branches
- **`mouse.sh`** - Mouse cell differentiation with 2 branches
- **`clonidine.sh`** - Clonidine perturbation with 2 branches
- **`trametinib.sh`** - Trametinib perturbation with 3 branches
- **`veres.sh`** - Pancreatic beta-cell differentiation with 11 branches


The scripts for the baseline single-branch SBM experiments include:

- **`mouse_single.sh`** - Mouse single branch
- **`clonidine_single.sh`** - Clonidine single branch
- **`trametinib_single.sh`** - Trametinib single branch
- **`lidar_single.sh`** - LiDAR single branch

**Before running experiments:**

1. Set `HOME_LOC` to the base path where BranchSBM is located and `ENV_PATH` to the directory where your environment is downloaded in the `.sh` files in `scripts/`
2. Create a path `BranchSBM/results` where the simulated trajectory figures and metrics will be saved. Also, create `BranchSBM/logs` where the training logs will be saved.
3. Activate the conda environment:
```
conda activate branchsbm
```
4. Login to wandb using `wandb login`

**Run experiment using `nohup` with the following commands:**

```
cd scripts

chmod lidar.sh

nohup ./lidar.sh > lidar.log 2>&1 &
```

Evaluation will run automatically after the specified number of rollouts `--num_rollouts` is finished. To see metrics, go to `results/<experiment>/metrics/` or the end of `logs/<experiment>.log`. 

For Clonidine, `x1_1` indicates the cell cluster that is sampled from for training and `x1_2` is the held-out cell cluster. For Trametinib, `x1_1` indicates the cell cluster that is sampled from for training, and `x1_2` and `x1_3` are the held-out cell clusters.

We report the following metrics for each of the clusters in our paper:
1. Maximum Mean Discrepancy (RBF-MMD) of simulated cell cluster with target cell cluster (same cell count).
2. 1-Wasserstein and 2-Wasserstein distances against the full cell population in the cluster.

## Overview of Outputs

**Training outputs are saved to experiment-specific directories:**

```
BranchSBM/results/
├── <DATE>_clonidine50D_branched/
│   └── figures/         # Figures of simulated 
│   └── metrics.csv         # JSON of metrics
```

**PyTorch Lightning automatically saves model checkpoints to:**

```
BranchSBM/scripts/lightning_logs/
├── <wandb-run-id>/
│   ├── checkpoints/
│   │   ├── epoch=N-step=M.ckpt     # Checkpoint 
```

**Training logs are saved in:**
```
entangled-cell/logs/
├── <DATE>_lidar_single_train.log
├── <DATE>_lidar_train.log
├── <DATE>_mouse_single_train.log
├── <DATE>_mouse_train.log
├── <DATE>_clonidine_single_train.log
├── <DATE>_clonidine50D_train.log
├── <DATE>_clonidine100D_train.log
├── <DATE>_clonidine150D_train.log
├── <DATE>_trametinib_single_train.log
├── <DATE>_trametinib_train.log
└── <DATE>_veres_train.log
```

## Available Experiments

### Branched Experiments (Multi-branch trajectories)

These experiments model cell differentiation or perturbation with multiple branches:

- **`mouse.sh`** - Mouse cell differentiation with 2 branches (GPU 0)
- **`trametinib.sh`** - Trametinib perturbation with 3 branches (GPU 1)
- **`lidar.sh`** - LiDAR trajectory data with 2 branches (GPU 2)
- **`clonidine.sh`** - Clonidine perturbation with 2 branches (GPU 3)

### Single-Branch Experiments (Control/baseline)

These are baseline experiments with single trajectories:

- **`mouse_single.sh`** - Mouse single trajectory (GPU 4)
- **`clonidine_single.sh`** - Clonidine single trajectory (GPU 5)
- **`trametinib_single.sh`** - Trametinib single trajectory (GPU 6)
- **`lidar_single.sh`** - LiDAR single trajectory (GPU 7)

## Running Scripts

### Run a single experiment

From the `scripts/` directory:

```bash
cd scripts
chmod +x mouse.sh
nohup ./mouse.sh > mouse.log 2>&1 &
```

### Run all branched experiments in parallel

```bash
nohup ./mouse.sh > mouse.log 2>&1 &
nohup ./trametinib.sh > trametinib.log 2>&1 &
nohup ./lidar.sh > lidar.log 2>&1 &
nohup ./clonidine.sh > clonidine.log 2>&1 &
```

### Run all single-branch experiments in parallel

```bash
nohup ./mouse_single.sh > mouse_single.log 2>&1 &
nohup ./clonidine_single.sh > clonidine_single.log 2>&1 &
nohup ./trametinib_single.sh > trametinib_single.log 2>&1 &
nohup ./lidar_single.sh > lidar_single.log 2>&1 &
```

### Run all experiments simultaneously

Each script is assigned to a different GPU, so you can run all 8 in parallel:

```bash
nohup ./mouse.sh > mouse.log 2>&1 &
nohup ./trametinib.sh > trametinib.log 2>&1 &
nohup ./lidar.sh > lidar.log 2>&1 &
nohup ./clonidine.sh > clonidine.log 2>&1 &
nohup ./mouse_single.sh > mouse_single.log 2>&1 &
nohup ./clonidine_single.sh > clonidine_single.log 2>&1 &
nohup ./trametinib_single.sh > trametinib_single.log 2>&1 &
nohup ./lidar_single.sh > lidar_single.log 2>&1 &
```

## Monitoring Training

Logs are saved in `./BranchSBM/logs/` with format `MM_DD_<experiment>_train.log`.

Each experiment logs to wandb with a unique run name:
- Branched experiments: `<dataset>_branched` (e.g., `mouse_branched`)
- Single experiments: `<dataset>_single` (e.g., `mouse_single`)

Visit your wandb dashboard to view training progress in real-time.

## Training Parameters

Default training parameters for each experiment:

| Parameter | LiDAR | Mouse Hematopoiesis scRNA | Clonidine (50 PCs) | Clonidine (100 PCs) | Clonidine (150 PCs) | Trametinib | Pancreatic β-Cell |
|---|---|---|---|---|---|---|---|
| branches | 2 | 2 | 2 | 2 | 2 | 3 | 11 |
| data dimension | 3 | 2 | 50 | 100 | 150 | 50 | 30 |
| batch size | 128 | 128 | 32 | 32 | 32 | 32 | 256 |
| λ_energy | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| λ_mass | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| λ_match | 1.0 × 10³ | 1.0 × 10³ | 1.0 × 10³ | 1.0 × 10³ | 1.0 × 10³ | 1.0 × 10³ | 1.0 × 10³ |
| λ_recons | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| λ_growth | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| V_t | LAND | LAND | RBF | RBF | RBF | RBF | RBF |
| RBF N_c | - | - | 150 | 300 | 300 | 150 | 300 |
| RBF κ | - | - | 1.5 | 2.0 | 3.0 | 1.5 | 3.0 |
| hidden dimension | 64 | 64 | 1024 | 1024 | 1024 | 1024 | 1024 |
| lr interpolant | 1.0 × 10⁻⁴ | 1.0 × 10⁻⁴ | 1.0 × 10⁻⁴ | 1.0 × 10⁻⁴ | 1.0 × 10⁻⁴ | 1.0 × 10⁻⁴ | 1.0 × 10⁻⁴ |
| lr velocity | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ |
| lr growth | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ | 1.0 × 10⁻³ |

To modify parameters, edit the corresponding `.sh` file.

## Training Pipeline

Each experiment runs through 4 stages:

1. **Stage 1: Geopath** - Train geodesic path interpolants
2. **Stage 2: Flow Matching** - Train continuous normalizing flows
3. **Stage 3: Growth** - Train growth networks for branches
4. **Stage 4: Joint** - Joint training of all components

Checkpoints are saved automatically and loaded between stages.
