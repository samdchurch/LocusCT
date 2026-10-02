#!/bin/bash
#SBATCH --job-name=merlin_resample_voxtell_scratch_grid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/resample_merlin_voxtell_scratch_%j.out
#SBATCH --error=logs/resample_merlin_voxtell_scratch_%j.err

# CPU-only preprocessing step for running our --from-scratch VoxTell
# checkpoints (trained on the 1.5x1.5x3.0mm/352x352x180 grid -- see
# resample_and_crop.py) against the new native ICLR2027_merlin_images tree
# (original, un-resampled 512x512xN volumes). Those checkpoints' sliding-
# window inference does no resampling of its own (see finetune_voxtell.py's
# module docstring), so native-resolution input would be a real spacing
# mismatch -- this resamples once, up front, into its own output folder, the
# same way nifti_resampled/ already exists for the main in-house dataset.
#
# Uses resample_merlin.py's --input-root/--output-root/--target-spacing/
# --target-shape overrides (added alongside its existing default 192^3-grid
# behavior, which submit_resample_merlin_192.sh -- if/when one exists --
# would use unchanged) to point at ICLR2027_merlin_images and VoxTell
# --from-scratch's own grid instead of the file's own defaults (merlin_data /
# merlin_data_resampled_192 / 2.0x2.0x3.0mm / 192^3).
#
# Output files already existing at the destination are skipped (see
# resample_merlin.py's process_file), so re-running after an interruption
# does NOT reprocess already-resampled files.
#
# Usage: sbatch submit_resample_merlin_voxtell_scratch_grid.sh
#   Smoke test first: sbatch submit_resample_merlin_voxtell_scratch_grid.sh --dry-run

DATA_ROOT=/path/to/data
INPUT_ROOT=$DATA_ROOT/merlinabdominalctdataset/merlin_data/ICLR2027_merlin_images
OUTPUT_ROOT=$DATA_ROOT/merlinabdominalctdataset/merlin_data/ICLR2027_merlin_images_resampled_352x352x180
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
  scripts/data_prep/resample_merlin.py \
    --input-root $INPUT_ROOT \
    --output-root $OUTPUT_ROOT \
    --target-spacing 1.5 1.5 3.0 \
    --target-shape 352 352 180 \
    --workers "$SLURM_CPUS_PER_TASK" \
    "$@"

echo "Job finished: $(date)"
