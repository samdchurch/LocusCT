#!/bin/bash
#SBATCH --job-name=nifti_resample_prune
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/resample_prune_%j.out
#SBATCH --error=logs/resample_prune_%j.err

# Per-file: resamples each native image to both nifti_resampled/
# (1.5x1.5x3.0mm, 352x352x192) and nifti_resampled_192/ (2.0x2.0x3.0mm,
# 192^3), then deletes the native original immediately if it's unreferenced
# by any train/val/test manifest -- interleaved so disk usage never needs
# to hold the full native dataset AND both full resampled grids at once
# (unlike submit_resample_migrate_and_192.sh +
# submit_delete_unreferenced_originals.sh, which assumed enough headroom
# for that two-phase approach).
#
# --confirm-delete is REQUIRED for this to actually free space -- without
# it, this only resamples (same disk-space problem as the two-step
# approach). Validate on a small subset first:
#   sbatch submit_resample_and_prune.sh --accession <accession> --confirm-delete
# before trusting it with the full dataset.
#
# Usage: sbatch submit_resample_and_prune.sh [extra scripts/data_prep/resample_and_prune.py args]
#   e.g. sbatch submit_resample_and_prune.sh --confirm-delete
#        sbatch submit_resample_and_prune.sh --dry-run

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
  scripts/data_prep/resample_and_prune.py --workers "$SLURM_CPUS_PER_TASK" \
    --audit-log resample_and_prune_log_${SLURM_JOB_ID}.jsonl \
    "$@"

echo "Job finished: $(date)"
