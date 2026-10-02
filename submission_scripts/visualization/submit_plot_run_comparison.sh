#!/bin/bash
#SBATCH --job-name=grounder_plot_comparison
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_plot_run_comparison.sh <run_dir1> <run_dir2> [...] [--labels l1 l2 ...] [--output out.png]
#   e.g. sbatch submit_plot_run_comparison.sh \
#          runs/h200_4gpu_30pct/ch16_bs3_lr1e-04_fixed352x352x192_gated_cross_attention_mw \
#          runs/h200_4gpu_10pct/ch16_bs3_lr1e-04_fixed352x352x192_gated_cross_attention_mw \
#          --labels 30pct 10pct --output outputs/viz/30pct_vs_10pct.png

REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  $SIF \
  scripts/visualization/plot_run_comparison.py "$@"
