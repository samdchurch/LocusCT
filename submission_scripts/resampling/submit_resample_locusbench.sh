#!/bin/bash
#SBATCH --job-name=locusbench_resample
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/resample_locusbench_%j.out
#SBATCH --error=logs/resample_locusbench_%j.err

# Resamples LocusBench-ED and LocusBench-Onc onto the same grid Grounder and
# finetuned VoxTell train on (1.5x1.5x3.0mm/352x352x180 by default), via
# scripts/data_prep/resample_locusbench.py. Only Grounder's own model and the
# finetuned-VoxTell checkpoint need this -- SAT/SegVol/BiomedParse/pretrained
# VoxTell each do their own on-the-fly preprocessing and read LocusBench's
# raw images/masks directly (see that script's own docstring).
#
# Any extra arguments are passed straight through, e.g. --split ED to
# resample only one split, or --force to reprocess existing outputs.
#
# Usage: sbatch submit_resample_locusbench.sh [extra resample_locusbench.py args...]
#   e.g. sbatch submit_resample_locusbench.sh --split ED

DATA_ROOT=/path/to/data
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/data_prep/resample_locusbench.py \
    --locusbench-root $DATA_ROOT/LocusBench \
    --workers "$SLURM_CPUS_PER_TASK" \
    --force \
    "$@"

echo "Job finished: $(date)"
