#!/bin/bash
#SBATCH --job-name=grounder_checkunused
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submission_scripts/misc/submit_check_unused_nifti_files.sh [nifti_dir]
#   e.g. sbatch submission_scripts/misc/submit_check_unused_nifti_files.sh
#        sbatch submission_scripts/misc/submit_check_unused_nifti_files.sh \
#            /path/to/data/inhouse_abdominal_ct/nifti_resampled_192

DATA_ROOT=/path/to/data
NIFTI_DIR=${1:-$DATA_ROOT/inhouse_abdominal_ct/nifti}
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/data_prep/check_unused_nifti_files.py \
    --nifti-dir "$NIFTI_DIR" \
    --output unused_nifti_files.json
