#!/bin/bash

HOME_LOC=/path/to/your/home/BranchSBM
ENV_LOC=/path/to/your/envs/branchsbm
SCRIPT_LOC=$HOME_LOC
LOG_LOC=$HOME_LOC/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='clonidine_single'
PYTHON_EXECUTABLE=$ENV_LOC/bin/python

# Set GPU device
export CUDA_VISIBLE_DEVICES=3

# ===================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_LOC

cd $HOME_LOC

$PYTHON_EXECUTABLE $SCRIPT_LOC/train.py \
    --epochs 100 \
    --run_name "clonidine50D_single" \
    --config_path "$SCRIPT_LOC/configs/clonidine_50Dsingle.yaml" \
    --batch_size 32 >> ${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}_train.log 2>&1

conda deactivate