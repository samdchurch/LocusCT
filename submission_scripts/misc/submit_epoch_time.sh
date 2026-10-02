#!/bin/bash
#SBATCH --job-name=grounder_epoch_time
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Reports scripts/diagnostics/epoch_time.py's mean/median/min/max epoch time for the
# four runs currently being compared: Grounder at 100pct/30pct data and VoxTell
# from-scratch at 30pct / fine-tuned from pretrained weights.
#
# Grounder run_names are computed by train.py's _build_run_name from configs/
# default.yaml's current defaults (see submit_h200_4gpu.sh/submit_h200_4gpu_30pct.sh) --
# if that config changes, the paths below need updating to match, since the actual
# on-disk directory name won't move with it. Update paths below to match if a run's
# location has moved (e.g. voxtell_finetune_l40s vs. voxtell_finetune_l40s_mw --
# see submit_finetune_voxtell_l40s.sh's own --multi-window-dependent routing).
#
# Usage: sbatch submit_epoch_time.sh

REPO=$HOME/grounder
SIF=$REPO/grounder.sif

RUN_NAME=ch16_bs3_lr1e-04_fixed352x352x192_gated_cross_attention_mw

declare -A RUNS=(
  ["Grounder 100pct"]="runs/h200_4gpu/$RUN_NAME/metrics.jsonl"
  ["Grounder 30pct"]="runs/h200_4gpu_30pct/$RUN_NAME/metrics.jsonl"
  ["VoxTell from-scratch 30pct"]="runs/voxtell_scratch_h200_4gpu_30pct/metrics.jsonl"
  ["VoxTell fine-tuned"]="runs/voxtell_finetune_l40s/metrics.jsonl"
)

for label in "Grounder 100pct" "Grounder 30pct" "VoxTell from-scratch 30pct" "VoxTell fine-tuned"; do
  echo "=== $label (${RUNS[$label]}) ==="
  singularity run \
    --pwd /workspace \
    --bind $REPO:/workspace \
    $SIF \
    scripts/diagnostics/epoch_time.py "${RUNS[$label]}"
  echo
done
