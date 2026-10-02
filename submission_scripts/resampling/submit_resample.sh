#!/bin/bash
#SBATCH --job-name=nifti_resample
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/resample_%j.out
#SBATCH --error=logs/resample_%j.err

# Any extra arguments are passed straight through to resample_masks.py --
# e.g. --force to re-resample masks already present in labels_resampled/
# onto a regenerated nifti_resampled/ image grid (default behavior only
# fills in missing files, it never overwrites existing ones).
#
# Usage: sbatch submit_resample.sh [extra resample_masks.py args...]
#   e.g. sbatch submit_resample.sh --force

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
  scripts/data_prep/resample_masks.py --workers "$SLURM_CPUS_PER_TASK" "$@"

echo "Job finished: $(date)"
