#!/bin/bash
#SBATCH --job-name=grounder_smoke_grpimg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Smoke-tests the group_by_image training path (GrounderImageGroupedDataset +
# UNet3D.forward's shared-encoder broadcast, see data/dataset.py and
# models/unet3d.py) against a tiny 8-sample slice of the real training
# manifest -- confirms the run completes and metrics.jsonl has sane
# (non-NaN, non-crazy) train_loss/train_dice. tests/ (submit_tests.sh)
# already covers unit-level correctness; this is the integration-level check
# from that change's verification plan -- a real forward/backward/optimizer
# step through the actual data pipeline and config, not synthetic tensors.
#
# Run both variants and diff their metrics.jsonl to compare against the old
# per-triplet path:
#   sbatch submit_smoke_group_by_image.sh        # group_by_image=true (default)
#   sbatch submit_smoke_group_by_image.sh false   # group_by_image=false, for comparison
#
# Usage: sbatch submit_smoke_group_by_image.sh [true|false]

GROUP_BY_IMAGE=${1:-true}

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
  train.py --config configs/default.yaml \
    --override model.text_encoder_name=$MODEL_DIR \
      data.group_by_image=$GROUP_BY_IMAGE \
      data.max_samples=8 \
      training.num_epochs=1 \
      checkpoint.output_dir=runs/smoke_grpimg_$GROUP_BY_IMAGE
