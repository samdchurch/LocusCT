#!/bin/bash
#SBATCH --job-name=nifti_resample_migrate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/resample_migrate_%j.out
#SBATCH --error=logs/resample_migrate_%j.err

# Two sequential steps against the existing scripts/data_prep/resample_and_crop.py:
#
# 1. Regenerates nifti_resampled/ IN PLACE with new params (1.5x1.5x3.0mm,
#    352x352x192) -- same spacing as its current actual content, but a
#    taller Z shape (192 vs the current 180), so dst.exists() would
#    otherwise skip every file untouched. Uses --force to overwrite every
#    existing file, not just fill in gaps. This is the grid
#    configs/default.yaml's data.image_dir points to.
#
# 2. Fills in nifti_resampled_192/ (2.0x2.0x3.0mm, 192^3), skipping any
#    files already resampled there (no --force -- safe to resume).
#
# Step 1 force-reprocesses the full dataset (not just new files), so this
# is budgeted well above the original single-grid jobs (10:00:00/8 cores) --
# adjust --time/--cpus-per-task to actual queue limits if needed.

DATA_ROOT=/path/to/data/inhouse_abdominal_ct
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

echo "--- Step 1/2: regenerating nifti_resampled/ (1.5x1.5x3.0mm, 352x352x192) ---"
singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/data_prep/resample_and_crop.py --workers "$SLURM_CPUS_PER_TASK" --force \
    --target-spacing 1.5 1.5 3.0 --target-shape 352 352 192 \
    --output-root $DATA_ROOT/nifti_resampled

echo "--- Step 2/2: filling in nifti_resampled_192/ (2.0x2.0x3.0mm, 192^3) ---"
singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/data_prep/resample_and_crop.py --workers "$SLURM_CPUS_PER_TASK" \
    --target-spacing 2.0 2.0 3.0 --target-shape 192 192 192 \
    --output-root $DATA_ROOT/nifti_resampled_192

echo "Job finished: $(date)"
