import sys
import os
import argparse
import copy
import time
import json

import torch.nn as nn
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchcfm.optimal_transport import OTPlanSampler

from parsers import parse_args
from train_utils import load_config, merge_config, generate_group_string, dataset_name2datapath, create_callbacks
from src.branchsbm import BranchSBM
from src.branch_flow_net_train import FlowNetTrainCell, FlowNetTrainLidar
from src.branch_flow_net_test import (
    FlowNetTestLidar, FlowNetTestMouse, FlowNetTestClonidine, FlowNetTestTrametinib, FlowNetTestVeres
)
from src.branch_interpolant_train import BranchInterpolantTrain
from src.branch_growth_net_train import GrowthNetTrain, GrowthNetTrainCell, GrowthNetTrainLidar, SequentialGrowthNetTrain
from src.networks.flow_mlp import VelocityNet
from src.networks.growth_mlp import GrowthNet
from src.networks.interpolant_mlp import GeoPathMLP
from src.utils import set_seed
from src.ema import EMA
from src.geo_metrics.metric_factory import DataManifoldMetric
from dataloaders.mouse_data import WeightedBranchedCellDataModule, SingleBranchCellDataModule
from dataloaders.three_branch_data import ThreeBranchTahoeDataModule
from dataloaders.clonidine_v2_data import ClonidineV2DataModule
from dataloaders.clonidine_single_branch import ClonidineSingleBranchDataModule
from dataloaders.trametinib_single import TrametinibSingleBranchDataModule
from dataloaders.lidar_data import WeightedBranchedLidarDataModule
from dataloaders.lidar_data_single import LidarSingleDataModule
from dataloaders.veres_leiden_data import WeightedBranchedVeresDataModule

def main(args: argparse.Namespace, seed: int, t_exclude: int) -> None:
    set_seed(seed)
    branches = args.branches

    skipped_time_points = [t_exclude] if t_exclude else []
    print("config path:")
    print(args.config_path)
    print("whiten")
    print(args.whiten)
    
    # Add date and time prefix to run name for distinguishable results
    current_datetime = time.strftime("%m_%d_%H%M", time.localtime())
    run_name_with_datetime = f"{current_datetime}_{args.run_name}"
    
    # Update args.run_name so test classes use the dated name
    args.run_name = run_name_with_datetime
    
    ### DATAMODULES
    
    ### DATAMODULES ###
    if args.data_name == "lidar":
        datamodule = WeightedBranchedLidarDataModule(args=args)
    elif args.data_name == "lidarsingle":
        datamodule = LidarSingleDataModule(args=args)
    elif args.data_name == "mouse":
        datamodule = WeightedBranchedCellDataModule(args=args)
    elif args.data_name == "mousesingle":
        datamodule = SingleBranchCellDataModule(args=args)
    elif args.data_name in ["clonidine50D", "clonidine100D", "clonidine150D"]:
        datamodule = ClonidineV2DataModule(args=args)  
    elif args.data_name == "clonidine50Dsingle":
        datamodule = ClonidineSingleBranchDataModule(args=args)
    elif args.data_name == "trametinib":
        datamodule = ThreeBranchTahoeDataModule(args=args)
    elif args.data_name == "trametinibsingle":
        datamodule = TrametinibSingleBranchDataModule(args=args)
    elif args.data_name == "veres":
        datamodule = WeightedBranchedVeresDataModule(args=args)
        branches = datamodule.num_branches
        print("number of branches:", branches)
    
    flow_nets = nn.ModuleList()
    geopath_nets = nn.ModuleList()
    growth_nets = nn.ModuleList()
        
    ##### initialize branched flow and growth networks #####
    for i in range(branches):
        flow_net = VelocityNet(
            dim=args.dim,
            hidden_dims=args.hidden_dims_flow,
            activation=args.activation_flow,
            batch_norm=False,
        )            
        geopath_net = GeoPathMLP(
            input_dim=args.dim,
            hidden_dims=args.hidden_dims_geopath,
            time_geopath=args.time_geopath,
            activation=args.activation_geopath,
            batch_norm=False,
        )
        
        if i == 0:
            growth_net = GrowthNet(
                dim=args.dim,
                hidden_dims=args.hidden_dims_growth,
                activation=args.activation_growth,
                batch_norm=False,
                negative=True
            )
        else: 
            growth_net = GrowthNet(
                dim=args.dim,
                hidden_dims=args.hidden_dims_growth,
                activation=args.activation_growth,
                batch_norm=False,
                negative=False
            )
        
        if args.ema_decay is not None:
            flow_net = EMA(model=flow_net, decay=args.ema_decay)
            geopath_net = EMA(model=geopath_net, decay=args.ema_decay)
            growth_net = EMA(model=growth_net, decay=args.ema_decay)
        
        flow_nets.append(flow_net)
        geopath_nets.append(geopath_net)
        growth_nets.append(growth_net)
        
    
    ot_sampler = (
        OTPlanSampler(method=args.optimal_transport_method)
        if args.optimal_transport_method != "None"
        else None
    )

    wandb.init(
        project="branchsbm",
        name=run_name_with_datetime,
        config=vars(args),
        dir=args.working_dir,
    )

    flow_matcher_base = BranchSBM(
        geopath_nets=geopath_nets,
        sigma=args.sigma,
        alpha=int(args.branchsbm),
    )

    ##### STAGE 1: Training of Geodesic Interpolants Beginning #####
    geopath_callbacks = create_callbacks(
        args, phase="geopath", data_type=args.data_type, run_id=wandb.run.id
    )
    
    # define state cost
    data_manifold_metric = DataManifoldMetric(
        args=args,
        skipped_time_points=skipped_time_points,
        datamodule=datamodule,
    )
    geopath_model = BranchInterpolantTrain(
        flow_matcher=flow_matcher_base,
        skipped_time_points=skipped_time_points,
        ot_sampler=ot_sampler,
        args=args,
        data_manifold_metric=data_manifold_metric
    )
    
    wandb_logger = WandbLogger(version=run_name_with_datetime)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=geopath_callbacks,
        accelerator=args.accelerator,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        default_root_dir=args.working_dir,
        gradient_clip_val=(1.0 if args.data_type == "image" else None),
    )
    
    if args.load_geopath_model_ckpt:
        best_model_path = args.load_geopath_model_ckpt
    else:
        trainer.fit(
            geopath_model,
            datamodule=datamodule,
        )
        
        best_model_path = geopath_callbacks[0].best_model_path
        
    geopath_model = BranchInterpolantTrain.load_from_checkpoint(best_model_path)

    flow_matcher_base.geopath_nets = geopath_model.geopath_nets

    ##### STAGE 1: Training of Geodesic Interpolants End #####

    ##### STAGE 2: Flow Matching Beginning #####
    flow_callbacks = create_callbacks(
        args,
        phase="flow",
        data_type=args.data_type,
        run_id=wandb.run.id,
        datamodule=datamodule,
    )
    
    if args.data_type == "lidar":
        FlowNetTrain = FlowNetTrainLidar
    else:
        FlowNetTrain = FlowNetTrainCell

    flow_train = FlowNetTrain(
        flow_matcher=flow_matcher_base,
        flow_nets=flow_nets,
        ot_sampler=ot_sampler,
        skipped_time_points=skipped_time_points,
        args=args,
    )

    # Reuse existing wandb run from Stage 1
    wandb_logger = WandbLogger(version=run_name_with_datetime)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=flow_callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accelerator=args.accelerator,
        logger=wandb_logger,
        default_root_dir=args.working_dir,
        gradient_clip_val=(1.0 if args.data_type == "image" else None),
        num_sanity_val_steps=(0 if args.data_type == "image" else None),
    )

    trainer.fit(
        flow_train, datamodule=datamodule, ckpt_path=args.resume_flow_model_ckpt
    )
    if args.data_type == "lidar":
        trainer.test(flow_train, datamodule=datamodule)
    ##### STAGE 2: Flow Matching End #####
    
    ##### STAGE 3: Training Growth Networks Beginning ####
    flow_nets = flow_train.flow_nets
    
    growth_callbacks = create_callbacks(
        args,
        phase="growth",
        data_type=args.data_type,
        run_id=wandb.run.id,
        datamodule=datamodule,
    )

    if args.data_type == "lidar":
        GrowthNetTrainClass = GrowthNetTrainLidar
    else:
        GrowthNetTrainClass = GrowthNetTrainCell
    
    growth_train = GrowthNetTrainClass(
        flow_nets = flow_nets,
        growth_nets = growth_nets,
        ot_sampler=ot_sampler,
        skipped_time_points=skipped_time_points,
        args=args,
        data_manifold_metric=data_manifold_metric,
        joint = False
    )

    # Reuse existing wandb run
    wandb_logger = WandbLogger(version=run_name_with_datetime)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=growth_callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accelerator=args.accelerator,
        logger=wandb_logger,
        default_root_dir=args.working_dir,
        gradient_clip_val=(1.0 if args.data_type == "image" else None),
        num_sanity_val_steps=(0 if args.data_type == "image" else None),
    )
    
    trainer.fit(
        growth_train, datamodule=datamodule, ckpt_path=None
    )
    
    # Load best checkpoint for testing
    best_growth_path = growth_callbacks[0].best_model_path
    if best_growth_path:
        print(f"Loading best growth model from: {best_growth_path}")
        if args.sequential:
            growth_train = SequentialGrowthNetTrain.load_from_checkpoint(
                best_growth_path,
                flow_nets=flow_nets,
                growth_nets=growth_nets,
                ot_sampler=ot_sampler,
                skipped_time_points=skipped_time_points,
                args=args,
                data_manifold_metric=data_manifold_metric,
                joint=False
            )
        else:
            growth_train = GrowthNetTrainClass.load_from_checkpoint(
                best_growth_path,
                flow_nets=flow_nets,
                growth_nets=growth_nets,
                ot_sampler=ot_sampler,
                skipped_time_points=skipped_time_points,
                args=args,
                data_manifold_metric=data_manifold_metric,
                joint=False
            )
        # Extract the trained flow_nets from the loaded checkpoint
        flow_nets = growth_train.flow_nets
        # Ensure flow_nets and growth_nets are ModuleList (not tuple)
        if isinstance(flow_nets, tuple):
            flow_nets = nn.ModuleList(flow_nets)
        if isinstance(growth_nets, tuple):
            growth_nets = nn.ModuleList(growth_nets)
    
    # Use appropriate test class based on data type
    if "lidar" in args.data_name.lower():
        test_model = FlowNetTestLidar(
            flow_nets = flow_nets,
            growth_nets = growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
            data_manifold_metric=data_manifold_metric,
            joint = False
        )
    elif "mouse" in args.data_name.lower():
        test_model = FlowNetTestMouse(
            flow_nets = flow_nets,
            growth_nets = growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
            data_manifold_metric=data_manifold_metric,
            joint = False
        )
    elif "clonidine" in args.data_name.lower():
        test_model = FlowNetTestClonidine(
            flow_matcher=flow_matcher_base,
            flow_nets=flow_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
        )
    elif "trametinib" in args.data_name.lower():
        test_model = FlowNetTestTrametinib(
            flow_matcher=flow_matcher_base,
            flow_nets=flow_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
        )
    elif "veres" in args.data_name.lower():
        test_model = FlowNetTestVeres(
            flow_nets = flow_nets,
            growth_nets = growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
            data_manifold_metric=data_manifold_metric,
            joint = False
        )
    else:
        # Default to growth_train test
        test_model = growth_train
    
    trainer.test(test_model, datamodule=datamodule)
    
    ##### STAGE 3: Training Growth Networks End ####
    
    ##### STAGE 4: Joint Training Beginning ####
    
    growth_nets = growth_train.growth_nets
    
    joint_callbacks = create_callbacks(
        args,
        phase="joint",
        data_type=args.data_type,
        run_id=wandb.run.id,
        datamodule=datamodule,
    )
    
    if args.sequential:
        joint_train = SequentialGrowthNetTrain(
            flow_nets = flow_nets,
            growth_nets = growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
            data_manifold_metric=data_manifold_metric,
            joint = True
        )
    else:
        if args.data_type == "lidar":
            GrowthNetTrainClass = GrowthNetTrainLidar
        else:
            GrowthNetTrainClass = GrowthNetTrainCell
            
        joint_train = GrowthNetTrainClass(
            flow_nets = flow_nets,
            growth_nets = growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
            data_manifold_metric=data_manifold_metric,
            joint = True
        )
    
    # Reuse existing wandb run
    wandb_logger = WandbLogger(version=run_name_with_datetime)

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=joint_callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accelerator=args.accelerator,
        logger=wandb_logger,
        default_root_dir=args.working_dir,
        gradient_clip_val=(1.0 if args.data_type == "image" else None),
        num_sanity_val_steps=(0 if args.data_type == "image" else None),
    )
    
    trainer.fit(
        joint_train, datamodule=datamodule, ckpt_path=None
    )
    
    # Load best checkpoint for testing
    best_joint_path = joint_callbacks[0].best_model_path
    if best_joint_path:
        print(f"Loading best joint model from: {best_joint_path}")
        if args.sequential:
            joint_train = SequentialGrowthNetTrain.load_from_checkpoint(
                best_joint_path,
                flow_nets=flow_nets,
                growth_nets=growth_nets,
                ot_sampler=ot_sampler,
                skipped_time_points=skipped_time_points,
                args=args,
                data_manifold_metric=data_manifold_metric,
                joint=True
            )
        else:
            joint_train = GrowthNetTrainClass.load_from_checkpoint(
                best_joint_path,
                flow_nets=flow_nets,
                growth_nets=growth_nets,
                ot_sampler=ot_sampler,
                skipped_time_points=skipped_time_points,
                args=args,
                data_manifold_metric=data_manifold_metric,
                joint=True
            )
        # Extract the trained flow_nets and growth_nets from the loaded checkpoint
        flow_nets = joint_train.flow_nets
        growth_nets = joint_train.growth_nets
        # Ensure flow_nets and growth_nets are ModuleList (not tuple)
        if isinstance(flow_nets, tuple):
            flow_nets = nn.ModuleList(flow_nets)
        if isinstance(growth_nets, tuple):
            growth_nets = nn.ModuleList(growth_nets)
    
    # Use appropriate test class based on data type
    if "lidar" in args.data_name.lower():
        test_model = FlowNetTestLidar(
            flow_nets = flow_nets,
            growth_nets = growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
            data_manifold_metric=data_manifold_metric,
            joint = True
        )
    elif "mouse" in args.data_name.lower():
        test_model = FlowNetTestMouse(
            flow_nets = flow_nets,
            growth_nets = growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
            data_manifold_metric=data_manifold_metric,
            joint = True
        )
    elif "clonidine" in args.data_name.lower():
        test_model = FlowNetTestClonidine(
            flow_matcher=flow_matcher_base,
            flow_nets=flow_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
        )
    elif "trametinib" in args.data_name.lower():
        test_model = FlowNetTestTrametinib(
            flow_matcher=flow_matcher_base,
            flow_nets=flow_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
        )
    elif "veres" in args.data_name.lower():
        test_model = FlowNetTestVeres(
            flow_nets = flow_nets,
            growth_nets = growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=skipped_time_points,
            args=args,
            data_manifold_metric=data_manifold_metric,
            joint = True
            )
    else:
        test_model = joint_train
        test_model = joint_train
    
    trainer.test(test_model, datamodule=datamodule)
    
    ##### STAGE 4: Joint Training End ####
    
    wandb.finish()
    
if __name__ == "__main__":
    args = parse_args()
    updated_args = copy.deepcopy(args)
    if args.config_path:
        config = load_config(args.config_path)
        updated_args = merge_config(updated_args, config)

    updated_args.group_name = generate_group_string()
    updated_args.data_path = dataset_name2datapath(
        updated_args.data_name, updated_args.working_dir
    )
    for seed in updated_args.seeds:
        if updated_args.t_exclude:
            for i, t_exclude in enumerate(updated_args.t_exclude):
                updated_args.t_exclude_current = t_exclude
                updated_args.seed_current = seed
                updated_args.gamma_current = updated_args.gammas[i]
                main(updated_args, seed=seed, t_exclude=t_exclude)
        else:
            updated_args.seed_current = seed
            updated_args.gamma_current = updated_args.gammas[0]
            main(updated_args, seed=seed, t_exclude=None)
