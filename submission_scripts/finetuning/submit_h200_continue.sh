#!/bin/bash
#SBATCH --job-name=grounder_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Continues normal training after submit_restart_text_proj.sh's one-time
# warm-start (--resume-partial + --init-text-proj) already ran. Unlike that
# script, this is a plain run with no special resume flags -- train.py's
# auto-resume already finds the latest checkpoint (including mid_epoch.pt,
# if training.checkpoint_every_n_steps is set) under checkpoint.output_dir
# and restores model/optimizer/scheduler/epoch state normally, since every
# checkpoint written to runs/h200_restart_full now shares the same architecture.
#
# Sized to match what submit_restart_text_proj.sh last ran with (3 GPUs, not
# a full 8-GPU node -- see that script's header for why). Bump
# --gres/--nproc_per_node/--cpus-per-task/--mem back up together if more
# H200_nvl GPUs are free now (check with
# `scontrol show node <name> | grep -E "CfgTRES|AllocTRES"`).
#
# Usage: sbatch submit_h200_continue.sh [checkpoint] [override_lr]
#   e.g. sbatch submit_h200_continue.sh runs/h200_restart_full/<run_name>/checkpoints/mid_epoch.pt 5e-4
# Omit checkpoint to let train.py's auto-resume find the latest one itself
# (mid_epoch.pt if present, else the latest epoch_*.pt) under
# checkpoint.output_dir -- pass one explicitly to resume from a specific
# point instead (e.g. best.pt, or an older epoch_*.pt).
# Omit override_lr to continue at whatever LR the resumed checkpoint's
# optimizer/scheduler state already has. Do NOT achieve this by editing
# configs/default.yaml's optimizer.lr instead -- a normal (strict) resume's
# optimizer.load_state_dict()/scheduler.load_state_dict() would silently
# overwrite that back to the checkpoint's saved value, and changing it also
# renames checkpoint.output_dir via _build_run_name, pointing auto-resume at
# an empty directory instead of runs/h200_restart_full's existing checkpoints.

CHECKPOINT=${1:-}
LR=${2:-}

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT, and the embedding_cache
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

EXTRA_ARGS=()
if [ -n "$CHECKPOINT" ]; then
  EXTRA_ARGS+=(--resume "$CHECKPOINT")
fi
if [ -n "$LR" ]; then
  EXTRA_ARGS+=(--override-lr "$LR")
fi

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
    "${EXTRA_ARGS[@]}" \
    --override model.text_encoder_name=$MODEL_DIR \
      checkpoint.output_dir=runs/h200_restart_full
