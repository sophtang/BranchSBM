#!/bin/bash

HOME_LOC=/path/to/your/home/BranchSBM
ENV_LOC=/path/to/your/envs/branchsbm
SCRIPT_LOC=$HOME_LOC
LOG_LOC=$HOME_LOC/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='lidar_single'
# set 3 have skip connection
PYTHON_EXECUTABLE=$ENV_LOC/bin/python

# Set GPU device
export CUDA_VISIBLE_DEVICES=2

# ===================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_LOC

cd $HOME_LOC

$PYTHON_EXECUTABLE $SCRIPT_LOC/train.py \
    --config_path "$SCRIPT_LOC/configs/lidar_single.yaml" \
    --run_name "${DATE}_${SPECIAL_PREFIX}" \
    --epochs 100 \
    --batch_size 128 >> ${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}_train.log 2>&1

conda deactivate