# [Branched Schrödinger Bridge Matching](https://arxiv.org/abs/2506.09007) (ICLR 2026) 🌳🧬

[**Sophia Tang**](https://sophtang.github.io/), [**Yinuo Zhang**](https://www.linkedin.com/in/yinuozhang98/), [**Alexander Tong**](https://www.alextong.net/) and [**Pranam Chatterjee**](https://www.chatterjeelab.com/)

![BranchSBM](assets/branchsbm_anim.gif)

This is the repository for [**Branched Schrödinger Bridge Matching**](https://arxiv.org/abs/2506.09007) (ICLR 2026) 🌳🧬. It is partially built on the [**Metric Flow Matching repo**](https://github.com/kkapusniak/metric-flow-matching) ([Kapusniak et al., 2024](https://arxiv.org/abs/2405.14780)).

Predicting how a population evolves between an initial and final state is central to many problems in generative modeling, from simulating perturbation responses to modelling cell fate decisions 🧫. Existing approaches, such as flow matching and Schrödinger Bridge Matching, effectively learn mappings between two distributions by modelling a single stochastic path. However, these methods are **inherently limited to unimodal transitions and cannot capture branched or divergent evolution from a common origin to multiple distinct outcomes.** 

A key challenge in trajectory matching is reconstructing multi-modal marginals, particularly when modes diverge along distinct dynamical paths . Existing Schrödinger bridge and flow matching frameworks approximate multi-modal distributions by simulating many *independent* particle trajectories, which are susceptible to mode collapse, with particles concentrating on dominant high-density modes or traversing only low-energy intermediate paths.

To address this, we introduce **Branched Schrödinger Bridge Matching (BranchSBM)** 🌳🧬, a novel framework that learns a set of diverging velocity fields to reconstruct multi-modal target distributions while simultaneously learning growth networks that allocate mass across branches. Guided by a time-dependent potential energy function Vt, BranchSBM captures diverging, energy-minimizing dynamics without requiring intermediate-time supervision and can generate the full branched evolution from a single initial sample.

🌟 We define the **Branched Generalized Schrödinger Bridge problem** and introduce BranchSBM, a novel matching framework that learns optimal branched trajectories from an initial distribution to multiple target distributions.

🌟 We derive the Branched Conditional Stochastic Optimal Control (CondSOC) problem as the sum of Unbalanced CondSOC objectives and leverage a multi-stage training algorithm to learn the optimal branching drift and growth fields that transport mass along a branched trajectory.

🌟 We demonstrate the unique capability of BranchSBM to model dynamic branching trajectories across various real-world problems, including 3D navigation over LiDAR manifolds, modelling differentiating single-cell population dynamics, and simulating heterogeneous cellular responses to drug perturbation.

# Experiments
Code and instructions to reproduce our results are provided in `/scripts/README`.

## LiDAR Experiment 🗻

As a proof of concept, we first evaluate BranchSBM for navigating branched paths along the surface of a three-dimensional LiDAR manifold, from an initial distribution to two distinct target distributions while remaining on low-altitude regions of the manifold.

![LiDAR Experiment](assets/lidar.png)

## Mouse Hematopoiesis and Pancreatic β-Cell Experiment 🧫

BranchSBM is uniquely positioned to model single-cell population dynamics where a homogeneous cell population (e.g., progenitor cells) differentiates into several distinct subpopulation branches, each of which independently undergoes growth dynamics. In this experiment, we demonstrate this capability on mouse hematopoiesis data and pancreatic β-cell differentiation data.

We evaluate BranchSBM on a mouse hematopoiesis scRNA-seq dataset containing three developmental time points representing progenitor cells differentiating into two terminal cell fates. Compared to a single-branch SBM, BranchSBM successfully learns distinct branching trajectories and accurately reconstructs intermediate cell states, demonstrating its ability to recover lineage bifurcation dynamics. 

![Mouse Experiment](assets/mouse.png)

We evaluate BranchSBM on a pancreatic β-cell differentiation dataset ([Veres et al., 2019](https://www.nature.com/articles/s41586-019-1168-5)) containing 51,274 cells collected across eight time points as human pluripotent stem cells differentiate into pancreatic β-like cells. Cells are projected into a 30-dimensional PCA space, and Leiden clustering is used to define 11 terminal cell populations at the final time point. 

BranchSBM is trained using only samples from the initial and final states, while intermediate distributions are inferred by learning trajectories constrained to the data manifold using an RBF state cost. BranchSBM not only reconstructs the multi-modal terminal distribution at the final time point with superior accuracy against all baselines, but also produces intermediate trajectories that are competitive with models trained directly on intermediate snapshots.

![Veres Experiment](assets/veres.png)

## Cell Perturbation Modelling Experiment 💉
Predicting the effects of perturbation on cell state dynamics is a crucial problem for therapeutic design. In this experiment, we leverage BranchSBM to model the **trajectories of a single cell line from a single homogeneous state to multiple heterogeneous states after a drug-induced perturbation**. We demonstrate that BranchSBM is capable of modeling high-dimensional gene expression data and learning branched trajectories that accurately reconstruct diverging perturbed cell populations.

We extract the data for a single cell line (A-549) under perturbation with Clonidine and Trametinib at 5 µL, selected based on cell abundance and response diversity from the Tahoe-100M dataset. 

For the Clonidine perturbation data, we show that **BranchSBM reconstructs the ground-truth distributions, capturing the location and spread of the dataset**, whereas single-branch SBM fails to differentiate cells in cluster 1 that differ from cluster 0 in higher-dimensional principal components. We also show that BranchSBM can simulate trajectories in high-dimensional state spaces by *scaling up to 150 PCs*.

![Clonidine Experiment](assets/clonidine.png)

We further show that BranchSBM can **scale beyond two branches by modeling the perturbed cell population of Trametinib-treated cells**, which diverge into *three distinct clusters*. We trained BranchSBM with three endpoints and single-branch SBM with one endpoint containing all three clusters on the top 50 PCs.

![Trametinib Experiment](assets/trametinib.png)


## Citation
If you find this repository helpful for your publications, please consider citing our paper:
```
@article{tang2026branchsbm,
  title={Branched Schrödinger Bridge Matching},
  author={Tang, Sophia and Zhang, Yinuo and Tong, Alexander and Chatterjee, Pranam},
  journal={14th International Conference on Learning Representations (ICLR 2026)},
  year={2026}
}
```
To use this repository, you agree to abide by the MIT License.