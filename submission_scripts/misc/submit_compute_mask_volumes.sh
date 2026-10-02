#!/bin/bash
#SBATCH --job-name=grounder_mask_volumes
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Computes the physical volume (mL/mm^3) of every ground-truth mask in the
# official ED and oncology test manifests (see scripts/compute_mask_volumes.py).
#
# Usage: sbatch submit_compute_mask_volumes.sh [output_dir]
#   e.g. sbatch submit_compute_mask_volumes.sh outputs/mask_volumes

OUTPUT_DIR=${1:-outputs/mask_volumes}

DATA_ROOT=/path/to/data/inhouse_abdominal_ct
ED_MASK_DIR=$DATA_ROOT/ED_TEST_SET_resampled
ONC_MASK_DIR=$DATA_ROOT/labels_resampled
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/compute_mask_volumes.py \
    --ed-mask-dir $ED_MASK_DIR \
    --onc-mask-dir $ONC_MASK_DIR \
    --output "$OUTPUT_DIR/results.json"
