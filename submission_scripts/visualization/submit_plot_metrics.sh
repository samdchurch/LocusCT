#!/bin/bash
#SBATCH --job-name=grounder_plot
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_plot_metrics.sh <results.json> [results2.json ...] [--labels l1 l2] [--output out.png]
#   e.g. sbatch submit_plot_metrics.sh runs/h200/eval_val.json runs/l40s/eval_val.json \
#                                      --labels h200 l40s --output comparison.png

REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  $SIF \
  scripts/visualization/plot_metrics.py "$@"
