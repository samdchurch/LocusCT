#!/bin/bash
#SBATCH --job-name=grounder_viz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_viz.sh <run_dir> [n_samples] [split]
#   e.g. sbatch submit_viz.sh runs/h200/ch16_bs4_lr2e-04_fixed352x352x192_cross_attention 300 val

RUN_DIR=${1:?Usage: sbatch submit_viz.sh <run_dir> [n_samples] [split]}
N_SAMPLES=${2:-300}
SPLIT=${3:-val}

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/visualization/visualize.py \
    --config configs/default.yaml \
    --run_dir $RUN_DIR \
    --n_samples $N_SAMPLES \
    --split $SPLIT
