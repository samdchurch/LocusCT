#!/bin/bash
#SBATCH --job-name=grounder_checkpadding
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Compares the fake-water-density-padding bug's fingerprint (fraction of
# voxels at exactly 0.0 HU) between the ED official test set and a sample of
# the training set, to test whether ED test images are disproportionately
# still-stale relative to c010922's fix.
#
# Usage: sbatch submit_check_ed_test_padding_bug.sh

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti_resampled
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

echo "=== ED official test set ==="
singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/archive/check_ed_test_padding_bug.py \
    --manifest official_splits/ed_official_test_data.json \
    --image-dir $IMAGE_DIR

echo ""
echo "=== Training set (first 40 unique images, for comparison) ==="
singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/archive/check_ed_test_padding_bug.py \
    --manifest official_splits/all_data_train.json \
    --image-dir $IMAGE_DIR \
    --n-samples 40
