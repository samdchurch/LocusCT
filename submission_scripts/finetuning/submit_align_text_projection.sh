#!/bin/bash
#SBATCH --job-name=grounder_align_proj
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_align_text_projection.sh <checkpoint> [steps] [lr]
#   e.g. sbatch submit_align_text_projection.sh runs/h200/checkpoints/best.pt 2000 1e-3
#
# Run this once after discovering the text_encoder.projection used to build
# embeddings_mmap was never persisted (see scripts/embedding/precompute_embeddings.py). Freezes
# the checkpoint's UNet and trains only a fresh projection layer to re-align
# with it. Output feeds scripts/embedding/precompute_embeddings.py --load-projection for
# regenerating every embedding cache consistently.
#
# steps counts optimizer updates, not raw samples -- scripts/recovery/align_text_projection.py
# accumulates gradients over 4 samples per update by default (--grad-accum-steps)
# to smooth out training.batch_size=1's per-sample noise. At ~1.5s/sample
# (observed), 2000 steps x 4 accum = 8000 samples ~ 3.3h; --time budgets extra.

CHECKPOINT=${1:?Usage: sbatch submit_align_text_projection.sh <checkpoint> [steps] [lr]}
STEPS=${2:-2000}
LR=${3:-1e-3}

DATA_ROOT=/path/to/data
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/recovery/align_text_projection.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --output text_projection_aligned.pt \
    --steps "$STEPS" \
    --lr "$LR" \
    --override model.text_encoder_name=$MODEL_DIR
