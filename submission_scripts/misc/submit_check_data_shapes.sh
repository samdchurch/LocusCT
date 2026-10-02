#!/bin/bash
#SBATCH --job-name=grounder_checkshapes
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Checks whether every (image, mask) pair in the train/val/test manifests
# has matching on-disk shapes, and reports the overall shape distribution
# for images and masks separately -- run after a resample-grid migration
# (e.g. submit_resample_migrate_and_192.sh) to confirm nifti_resampled/ and
# labels_resampled/ actually agree with each other, since that migration
# only regenerates images and has no mask-side counterpart.
#
# Usage: sbatch submit_check_data_shapes.sh

DATA_ROOT=/path/to/data/inhouse_abdominal_ct
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/diagnostics/check_data_shapes.py \
    --manifest official_splits/all_data_train.json \
      official_splits/curated_ed_onc_val_data.json \
      official_splits/all_test_data.json \
    --image-dir $DATA_ROOT/nifti_resampled \
    --mask-dir $DATA_ROOT/labels_resampled \
    --out logs/check_data_shapes.json
