#!/bin/bash
#SBATCH --job-name=grounder_profile_dl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

DATA_ROOT=/path/to/data/inhouse_abdominal_ct
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  profile_data_loading.py --config configs/default.yaml \
    --split train --n_batches 200 --warmup 10 \
    --out logs/profile_data_loading.json
