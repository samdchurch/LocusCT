#!/bin/bash
#SBATCH --job-name=grounder_restart_l40s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same one-time restart as submit_restart_text_proj.sh (see that script's
# header for the full --resume-partial/--init-text-proj explanation), sized
# for a full L40S node instead of waiting on a scarce H200_nvl node. 4 GPUs,
# not 8 -- the largest L40S node config on this cluster (<node> in
# `sinfo -p gpu -N -o "%N %G"`) tops out at 4; requesting 8 on one node fails
# at submission ("Requested node configuration is not available", not a queue
# wait -- SLURM rejects it outright since no matching node exists).
# batch_size=1 pinned explicitly (matches configs/default.yaml's current
# default, but stated here so this run doesn't silently change if that
# default ever does); lr lowered from the base config since this is warm-
# starting an already-trained UNet plus a freshly-seeded text_proj, not
# training from scratch -- default here is 1/4 of configs/default.yaml's
# optimizer.lr (2.0e-4 -> 5.0e-5), adjust via the third argument if needed.
#
# Usage: sbatch submit_restart_l40s.sh <old_checkpoint> [text_proj_init] [lr]
#   e.g. sbatch submit_restart_l40s.sh runs/h200/<run_name>/checkpoints/best.pt

CHECKPOINT=${1:?Usage: sbatch submit_restart_l40s.sh <old_checkpoint> [text_proj_init] [lr]}
TEXT_PROJ_INIT=${2:-text_projection_aligned.pt}
LR=${3:-5.0e-5}

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$REPO/logs/nccl_%j_rank%r.log

# Derived from the job ID, not hardcoded -- this job isn't --exclusive, so a
# fixed port collides (EADDRINUSE) with any other torch.distributed.run job
# sharing the node, including another of your own.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env NCCL_DEBUG=INFO \
  --env NCCL_DEBUG_FILE=$REPO/logs/nccl_%j_rank%r.log \
  --env CC=gcc \
  $SIF \
  -m torch.distributed.run --nproc_per_node=4 --master_port=$MASTER_PORT \
  train.py --config configs/default.yaml \
    --resume "$CHECKPOINT" \
    --resume-partial \
    --init-text-proj "$TEXT_PROJ_INIT" \
    --override model.text_encoder_name=$MODEL_DIR \
      training.batch_size=1 \
      optimizer.lr=$LR \
      checkpoint.output_dir=runs/l40s_restart
