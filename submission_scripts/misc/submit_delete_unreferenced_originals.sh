#!/bin/bash
#SBATCH --job-name=grounder_prune_originals
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Dry-run by default (no files deleted, just a JSON report of what would be
# removed) -- run submission_scripts/resampling/submit_resample_migrate_and_192.sh
# FIRST and confirm it succeeded before relying on this.
#
# Usage:
#   sbatch submission_scripts/misc/submit_delete_unreferenced_originals.sh                   # dry run
#   sbatch submission_scripts/misc/submit_delete_unreferenced_originals.sh --confirm-delete   # actually delete
# Any args are forwarded to scripts/data_prep/delete_unreferenced_originals.py.

DATA_ROOT=/path/to/data
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/data_prep/delete_unreferenced_originals.py \
    --audit-log deleted_nifti_originals_${SLURM_JOB_ID}.json \
    "$@"
