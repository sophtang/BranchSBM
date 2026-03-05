#!/bin/bash

HOME_LOC=/path/to/your/home/BranchSBM
ENV_LOC=/path/to/your/envs/branchsbm
SCRIPT_LOC=$HOME_LOC
LOG_LOC=$HOME_LOC/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='veres'
PYTHON_EXECUTABLE=$ENV_LOC/bin/python

# Set GPU device
export CUDA_VISIBLE_DEVICES=7

# ===================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_LOC

cd $HOME_LOC

$PYTHON_EXECUTABLE $SCRIPT_LOC/train.py \
    --epochs 100 \
    --run_name ${DATE}_${SPECIAL_PREFIX} \
    --min_cells 100 \
    --config $SCRIPT_LOC/configs/veres.yaml >> ${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}_train.log 2>&1

conda deactivate
