import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train BranchSBM")
    
    parser.add_argument("--seed", default=2, type=int)

    parser.add_argument(
        "--config_path", type=str, 
        default='', 
        help="Path to config file"
    )
    ####### ITERATES IN THE CODE #######
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds to iterate over",
    )
    parser.add_argument(
        "--t_exclude",
        nargs="+",
        type=int,
        default=None,
        help="Time points to exclude (iterating over)",
    )
    ####################################

    parser.add_argument(
        "--working_dir",
        type=str,
        default="path/to/your/home/BranchSBM",
        help="Working directory",
    )
    parser.add_argument(
        "--resume_flow_model_ckpt",
        type=str,
        default=None,
        help="Path to the flow model to resume training",
    )
    parser.add_argument(
        "--resume_growth_model_ckpt",
        type=str,
        default=None,
        help="Path to the flow model to resume training",
    )
    parser.add_argument(
        "--load_geopath_model_ckpt",
        type=str,
        default=None,
        help="Path to the geopath model to resume training",
    )
    parser.add_argument(
        "--sequential",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sequential training for multi-timepoint data",
    )
    parser.add_argument(
        "--discard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Discard small clusters instead of merging them in Leiden clustering",
    )
    parser.add_argument(
        "--pseudo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pseudotime-based clustering for Weinreb data instead of Leiden on t=2",
    )
    parser.add_argument(
        "--branches",
        type=int,
        default=2,
        help="Number of branches",
    )
    parser.add_argument(
        "--metric_clusters",
        type=int,
        default=3,
        help="Number of metric clusters",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for Leiden clustering",
    )

    ######### DATASETS #################
    parser = datasets_parser(parser)
    ####################################

    ######### IMAGE DATASETS ###########
    parser = image_datasets_parser(parser)
    ####################################

    ######### METRICS ##################
    parser = metric_parser(parser)
    ####################################

    ######### General Training #########
    parser = general_training_parser(parser)
    ####################################

    ######### Training GeoPath Network ####
    parser = geopath_network_parser(parser)
    ####################################

    ######### Training Flow Network ####
    parser = flow_network_parser(parser)
    ####################################
    
    parser = growth_network_parser(parser)

    return parser.parse_args()


def datasets_parser(parser):
    parser.add_argument("--dim", type=int, default=3, help="Dimension of data")

    parser.add_argument(
        "--data_type",
        type=str,
        default="lidar",
        help="Type of data, now wither scrna or one of toys",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="lidar data path",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="lidar",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--whiten",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whiten the data",
    )
    parser.add_argument(
        "--min_cells",
        type=int,
        default=500,
        help="Minimum cells per cluster for Leiden clustering",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of neighbors for KNN graph in Leiden clustering",
    )
    parser.add_argument(
        "--pseudotime_threshold",
        type=float,
        default=0.6,
        help="Pseudotime threshold for terminal cells (only used when --pseudo is True)",
    )
    parser.add_argument(
        "--terminal_neighbors",
        type=int,
        default=20,
        help="Number of neighbors for terminal cell clustering (only used when --pseudo is True)",
    )
    parser.add_argument(
        "--terminal_resolution",
        type=float,
        default=0.2,
        help="Resolution for terminal cell Leiden clustering (only used when --pseudo is True)",
    )
    parser.add_argument(
        "--n_dcs",
        type=int,
        default=10,
        help="Number of diffusion components for DPT (only used when --pseudo is True)",
    )
    parser.add_argument(
        "--initial_neighbors",
        type=int,
        default=30,
        help="Number of neighbors for initial kNN graph (only used when --pseudo is True)",
    )
    parser.add_argument(
        "--initial_resolution",
        type=float,
        default=1.0,
        help="Resolution for initial Leiden clustering (only used when --pseudo is True)",
    )
    return parser


def image_datasets_parser(parser):
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Size of the image",
    )
    parser.add_argument(
        "--x0_label",
        type=str,
        default="dog",
        help="Label for x0",
    )
    parser.add_argument(
        "--x1_label",
        type=str,
        default="cat",
        help="Label for x1",
    )
    return parser


def metric_parser(parser):
    parser.add_argument(
        "--branchsbm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If branched SBM",
    )
    parser.add_argument(
        "--n_centers",
        type=int,
        default=100,
        help="Number of centers for RBF network",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Kappa parameter for RBF network",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.001,
        help="Rho parameter in Riemanian Velocity Calculation",
    )
    parser.add_argument(
        "--velocity_metric",
        type=str,
        default="rbf",
        help="Metric for velocity calculation",
    )
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[0.2, 0.2],
        help="Gamma parameter in Riemanian Velocity Calculation",
    )
    
    parser.add_argument(
        "--metric_epochs",
        type=int,
        default=100,
        help="Number of epochs for metric learning",
    )
    parser.add_argument(
        "--metric_patience",
        type=int,
        default=20,
        help="Patience for metric learning",
    )
    parser.add_argument(
        "--metric_lr",
        type=float,
        default=1e-2,
        help="Learning rate for metric learning",
    )
    parser.add_argument(
        "--alpha_metric",
        type=float,
        default=1.0,
        help="Alpha parameter for metric learning",
    )

    return parser


def general_training_parser(parser):
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--optimal_transport_method",
        type=str,
        default="exact",
        help="Use optimal transport in CFM training",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=None,
        help="Decay for EMA",
    )
    parser.add_argument(
        "--split_ratios",
        nargs=2,
        type=float,
        default=[0.9, 0.1],
        help="Split ratios for training/validation data in CFM training",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Training accelerator"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name for the wandb run"
    )
    parser.add_argument(
        "--sim_num_steps",
        type=int,
        default=1000,
        help="Number of steps in simulation",
    )
    return parser


def geopath_network_parser(parser):
    parser.add_argument(
        "--manifold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If use data manifold metric",
    )
    parser.add_argument(
        "--patience_geopath",
        type=int,
        default=50,
        help="Patience for training geopath model",
    )
    parser.add_argument(
        "--hidden_dims_geopath",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for GeoPath model training",
    )
    parser.add_argument(
        "--time_geopath",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use time in GeoPath model",
    )
    parser.add_argument(
        "--activation_geopath",
        type=str,
        default="selu",
        help="Activation function for GeoPath",
    )
    parser.add_argument(
        "--geopath_optimizer",
        type=str,
        default="adam",
        help="Optimizer for GeoPath training",
    )
    parser.add_argument(
        "--geopath_lr",
        type=float,
        default=1e-4,
        help="Learning rate for GeoPath training",
    )
    parser.add_argument(
        "--geopath_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for GeoPath training",
    )
    parser.add_argument(
        "--mmd_weight",
        type=float,
        default=0.1,
        help="Weight for MMD loss at intermediate timepoints (only used when >2 timepoints)",
    )
    return parser


def flow_network_parser(parser):
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="Sigma parameter for CFM (variance)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping in CFM training",
    )
    parser.add_argument(
        "--hidden_dims_flow",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for CFM training",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=10,
        help="Check validation every N epochs during CFM training",
    )
    parser.add_argument(
        "--activation_flow",
        type=str,
        default="selu",
        help="Activation function for CFM",
    )
    parser.add_argument(
        "--flow_optimizer",
        type=str,
        default="adamw",
        help="Optimizer for GeoPath training",
    )
    parser.add_argument(
        "--flow_lr",
        type=float,
        default=1e-3,
        help="Learning rate for GeoPath training",
    )
    parser.add_argument(
        "--flow_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for GeoPath training",
    )
    return parser

def growth_network_parser(parser):
    parser.add_argument(
        "--patience_growth",
        type=int,
        default=5,
        help="Patience for early stopping in CFM training",
    )
    parser.add_argument(
        "--time_growth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use time in GeoPath model",
    )
    parser.add_argument(
        "--hidden_dims_growth",
        nargs="+",
        type=int,
        default=[64, 64, 64],
        help="Dimensions of hidden layers for growth net training",
    )
    parser.add_argument(
        "--activation_growth",
        type=str,
        default="tanh",
        help="Activation function for CFM",
    )
    parser.add_argument(
        "--growth_optimizer",
        type=str,
        default="adamw",
        help="Optimizer for GeoPath training",
    )
    parser.add_argument(
        "--growth_lr",
        type=float,
        default=1e-3,
        help="Learning rate for GeoPath training",
    )
    parser.add_argument(
        "--growth_weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for GeoPath training",
    )
    parser.add_argument(
        "--lambda_energy",
        type=float,
        default=1.0,
        help="Weight for energy loss",
    )
    parser.add_argument(
        "--lambda_mass",
        type=float,
        default=100.0,
        help="Weight for mass loss",
    )
    parser.add_argument(
        "--lambda_match",
        type=float,
        default=1000.0,
        help="Weight for matching loss",
    )
    parser.add_argument(
        "--lambda_recons",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss",
    )
    return parser