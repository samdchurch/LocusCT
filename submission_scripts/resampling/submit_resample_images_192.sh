#!/bin/bash
#SBATCH --job-name=nifti_resample_192
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/resample_images_192_%j.out
#SBATCH --error=logs/resample_images_192_%j.err

# Resamples CT images to 2.0x2.0x3.0mm spacing, 192^3 voxels, into their own
# nifti_resampled_192/ folder -- does not touch or overwrite the
# nifti_resampled/ output. Run submit_resample_masks_192.sh after this
# finishes so masks get resampled onto this same grid.
#
# Superseded by submit_resample_migrate_and_192.sh, which does this same
# step plus regenerating nifti_resampled/ in one job -- kept as a standalone
# script for rerunning just this grid on its own.

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
  scripts/data_prep/resample_and_crop.py --workers "$SLURM_CPUS_PER_TASK" \
    --target-spacing 2.0 2.0 3.0 --target-shape 192 192 192 \
    --output-root $DATA_ROOT/nifti_resampled_192

echo "Job finished: $(date)"
