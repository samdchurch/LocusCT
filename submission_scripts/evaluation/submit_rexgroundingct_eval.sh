#!/bin/bash
#SBATCH --job-name=grounder_rexval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_rexgroundingct_eval.sh <checkpoint> [output_dir]
#   e.g. sbatch submit_rexgroundingct_eval.sh runs/h200/checkpoints/best.pt outputs/eval/rexgroundingct_val

CHECKPOINT=${1:?Usage: sbatch submit_rexgroundingct_eval.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/rexgroundingct_val}

DATA_ROOT=/path/to/data
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_rexgroundingct_val.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --data-root $DATA_ROOT \
    --rexrank-eval-script $DATA_ROOT/ReXGroundingCT/rexrank_eval.py \
    --output-dir "$OUTPUT_DIR" \
    --override model.text_encoder_name=$MODEL_DIR \
      training.batch_size=1
