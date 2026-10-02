#!/bin/bash
#SBATCH --job-name=grounder_nii_benchmark
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

DATA_ROOT=/path/to/data/inhouse_abdominal_ct
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  make_nii_benchmark.py --n 1000 --workers 8
