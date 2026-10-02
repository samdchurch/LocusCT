#!/bin/bash
#SBATCH --job-name=grounder_restart
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# One-time restart after the text_proj architecture change (see
# models/cross_attention.py's per-stage text_proj, scripts/embedding/precompute_embeddings.py's
# raw-hidden-state cache format, and train.py's --resume-partial/--init-text-proj).
# Warm-starts the UNet conv/res backbone from the old checkpoint (--resume-partial,
# since text_proj.* keys are new -- everything else carries over by matching shape)
# and seeds every cross-attention stage's text_proj from the already-aligned
# projection (--init-text-proj) instead of a random init. Optimizer/scheduler/epoch
# do NOT carry over in this mode -- training restarts its bookkeeping from epoch 0,
# so this writes to its own runs/h200_restart_full rather than the original run's
# directory -- resuming at epoch 0 into the same dir would tangle its checkpoints
# (including its own best.pt) in with the original run's and could overwrite them.
#
# Sized for 3 GPUs (not --exclusive) rather than a full 8-GPU node: at the
# time this was written, no h200_nvl node had 8 free, but <node> had 3
# free (check with `scontrol show node <name> | grep -E "CfgTRES|AllocTRES"`
# -- CfgTRES minus AllocTRES gres/gpu is what's actually free). Bump
# --gres/--nproc_per_node/--cpus-per-task/--mem back up together if a full
# node opens up later; this is a one-time restart, not the main training config.
#
# Usage: sbatch submit_restart_text_proj.sh <old_checkpoint> [text_proj_init]
#   e.g. sbatch submit_restart_text_proj.sh runs/h200/<run_name>/checkpoints/best.pt

CHECKPOINT=${1:?Usage: sbatch submit_restart_text_proj.sh <old_checkpoint> [text_proj_init]}
TEXT_PROJ_INIT=${2:-text_projection_aligned.pt}

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
  -m torch.distributed.run --nproc_per_node=3 --master_port=$MASTER_PORT \
  train.py --config configs/default.yaml \
    --resume "$CHECKPOINT" \
    --resume-partial \
    --init-text-proj "$TEXT_PROJ_INIT" \
    --override model.text_encoder_name=$MODEL_DIR \
      checkpoint.output_dir=runs/h200_restart_full
