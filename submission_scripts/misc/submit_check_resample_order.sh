#!/bin/bash
#SBATCH --job-name=grounder_checkorder
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_check_resample_order.sh [accession] [series]
#   e.g. sbatch submit_check_resample_order.sh CASE0000000 2

ACCESSION=${1:-CASE0000000}
SERIES=${2:-2}

DATA_ROOT=/path/to/data
RAW_ROOT=$DATA_ROOT/inhouse_abdominal_ct/nifti
RESAMPLED_ROOT=$DATA_ROOT/inhouse_abdominal_ct/nifti_resampled
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/archive/check_resample_order.py \
    --accession "$ACCESSION" \
    --series "$SERIES" \
    --raw-root $RAW_ROOT \
    --resampled-root $RESAMPLED_ROOT
