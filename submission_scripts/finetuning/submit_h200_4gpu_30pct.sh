#!/bin/bash
#SBATCH --job-name=grounder_h200_4gpu_30pct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same run as submit_h200_4gpu.sh (352x352x192 default grid, lr=1e-4, no
# warmup, batch_size=3, 4x H200 DDP, trains until Macro Hit Rate early-stops)
# but on official_splits/all_data_train_30pct.json instead of the full
# training manifest. checkpoint.output_dir is its own base (not
# runs/h200_4gpu) since _build_run_name's auto-generated subdirectory name
# doesn't encode which train_manifest was used -- sharing a base with the
# full-data run would let --resume's auto-discovery pick up a checkpoint
# trained on the wrong data.
#
# Usage: sbatch submit_h200_4gpu_30pct.sh

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derived from the job ID, not hardcoded -- avoids EADDRINUSE against a
# leftover process from a previous job on this node (this job is
# --exclusive, so it won't collide with another concurrent job's port).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $SIF \
  -m torch.distributed.run --nproc_per_node=4 --master_port=$MASTER_PORT \
  train.py --config configs/default.yaml \
    --override model.text_encoder_name=$MODEL_DIR \
      checkpoint.output_dir=runs/h200_4gpu_30pct \
      data.train_manifest=official_splits/all_data_train_30pct.json \
      training.batch_size=3 \
      optimizer.lr=1e-4 \
      scheduler.warmup_epochs=0
