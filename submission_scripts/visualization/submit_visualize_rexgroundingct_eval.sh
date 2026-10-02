#!/bin/bash
#SBATCH --job-name=grounder_rexviz
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_visualize_rexgroundingct_eval.sh <eval_dir> [n_samples]
#   e.g. sbatch submit_visualize_rexgroundingct_eval.sh outputs/eval/rexgroundingct_val 200

EVAL_DIR=${1:-outputs/eval/rexgroundingct_val}
N_SAMPLES=${2:-100}

DATA_ROOT=/path/to/data
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/visualization/visualize_rexgroundingct_eval.py \
    --eval-dir "$EVAL_DIR" \
    --data-root $DATA_ROOT \
    --n-samples $N_SAMPLES
