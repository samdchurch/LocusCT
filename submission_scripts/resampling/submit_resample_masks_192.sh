#!/bin/bash
#SBATCH --job-name=mask_resample_192
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/resample_masks_192_%j.out
#SBATCH --error=logs/resample_masks_192_%j.err

# Resamples masks onto the 192^3 image grid produced by
# submit_resample_images_192.sh -- run that first so the reference images
# exist. Writes to their own labels_resampled_192/ and
# ED_TEST_SET_resampled_192/ folders, leaving the existing
# labels_resampled/ output untouched.

DATA_ROOT=/path/to/data/inhouse_abdominal_ct
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
  scripts/data_prep/resample_masks.py --workers "$SLURM_CPUS_PER_TASK" \
    --images-root $DATA_ROOT/nifti_resampled_192 \
    --output-root $DATA_ROOT/labels_resampled_192 \
    --ed-test-output-root $DATA_ROOT/ED_TEST_SET_resampled_192

echo "Job finished: $(date)"
