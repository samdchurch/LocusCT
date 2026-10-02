#!/bin/bash
#SBATCH --job-name=grounder_viz_train
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_visualize_train_samples.sh [n] [seed]
#   e.g. sbatch submit_visualize_train_samples.sh 20 0

N=${1:-20}
SEED=${2:-0}

DATA_ROOT=/path/to/data
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/visualization/visualize_train_samples.py \
    --image-dir $DATA_ROOT/inhouse_abdominal_ct/nifti \
    --mask-dir $DATA_ROOT/inhouse_abdominal_ct/ALL_LABELS \
    --n "$N" \
    --seed "$SEED" \
    --output-dir outputs/viz/train_samples
