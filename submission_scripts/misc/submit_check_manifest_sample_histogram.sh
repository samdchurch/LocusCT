#!/bin/bash
#SBATCH --job-name=grounder_checkhist
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_check_manifest_sample_histogram.sh <manifest> <mask_id> [image_dir] [mask_dir]
#   e.g. sbatch submit_check_manifest_sample_histogram.sh official_splits/all_data_train.json \
#            CASE0000000/mask_7_39_1.nii.gz

MANIFEST=${1:?Usage: sbatch submit_check_manifest_sample_histogram.sh <manifest> <mask_id> [image_dir] [mask_dir]}
MASK_ID=${2:?Usage: sbatch submit_check_manifest_sample_histogram.sh <manifest> <mask_id> [image_dir] [mask_dir]}

DATA_ROOT=/path/to/data
IMAGE_DIR=${3:-$DATA_ROOT/inhouse_abdominal_ct/nifti_resampled}
MASK_DIR=${4:-$DATA_ROOT/inhouse_abdominal_ct/labels_resampled}
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/archive/check_manifest_sample_histogram.py \
    --manifest "$MANIFEST" \
    --mask-id "$MASK_ID" \
    --image-dir $IMAGE_DIR \
    --mask-dir $MASK_DIR
