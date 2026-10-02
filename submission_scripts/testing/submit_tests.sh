#!/bin/bash
#SBATCH --job-name=grounder_pytest
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the full pytest suite (tests/) inside grounder.sif -- CPU-only, no
# GPU/data-root binds needed (every test builds tiny tensors/fixtures itself,
# see tests/conftest.py and tests/test_dataset.py's tmp_path-based manifest).
#
# Usage: sbatch submit_tests.sh [extra pytest args...]
#   e.g. sbatch submit_tests.sh
#        sbatch submit_tests.sh tests/test_unet3d.py -k voxtell

REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

# Default to the whole tests/ dir; any args passed to this script (a specific
# file, -k filter, etc.) replace that default rather than adding to it.
ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then
  ARGS=(tests/)
fi

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  $SIF \
  -m pytest -v "${ARGS[@]}"
