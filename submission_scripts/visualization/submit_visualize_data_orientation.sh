#!/bin/bash
#SBATCH --job-name=grounder_orientviz
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_visualize_data_orientation.sh [n_samples] [output_dir]
#   e.g. sbatch submit_visualize_data_orientation.sh 8 orientation_check

N_SAMPLES=${1:-5}
OUTPUT_DIR=${2:-outputs/viz/orientation_check}

DATA_ROOT=/path/to/data
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/visualization/visualize_data_orientation.py \
    --config configs/default.yaml \
    --n-samples $N_SAMPLES \
    --output-dir "$OUTPUT_DIR" \
    --override model.text_encoder_name=$MODEL_DIR
