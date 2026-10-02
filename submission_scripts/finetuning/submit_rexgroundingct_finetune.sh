#!/bin/bash
#SBATCH --job-name=grounder_rexft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Fine-tunes an already-trained Grounder checkpoint on ReXGroundingCT alone
# (official_splits/ReXGroundingCT_train.json), validating each epoch against
# official_splits/ReXGroundingCT_val.json -- see configs/rexgroundingct_finetune.yaml
# for manifest/path/hyperparameter details. Warm-starts model weights from
# CHECKPOINT via --resume-partial (strict=False load; optimizer/scheduler/
# epoch/best_dice are NOT restored) rather than a plain --resume, since this
# starts a new training phase on a different, much smaller dataset with its
# own LR schedule and epoch budget -- not a continuation of the source run's
# bookkeeping. Writes to its own checkpoint.output_dir
# (runs/rexgroundingct_finetune) so it can't collide with the source run's
# checkpoints.
#
# After training, score the resulting checkpoint on ReXGroundingCT_val.json
# with the official ReXrank metric via submit_rexgroundingct_eval.sh -- it
# already defaults to this val manifest, no separate eval script needed here.
#
# Sized to match submit_h200_continue.sh (3x H200_nvl, not a full 8-GPU node
# -- see that script's header for why free GPUs may be limited). Resize
# --gres/--nproc_per_node/--cpus-per-task/--mem together if a different
# allocation is free/needed -- GPU memory needs come from batch_size/model
# size, not dataset size, so this doesn't need to shrink just because
# ReXGroundingCT_train.json (~7.7k samples) is much smaller than the full
# corpus.
#
# Usage: sbatch submit_rexgroundingct_finetune.sh <checkpoint> [override_lr]
#   e.g. sbatch submit_rexgroundingct_finetune.sh runs/h200_restart_full/<run_name>/checkpoints/best.pt
# Omit override_lr to use configs/rexgroundingct_finetune.yaml's own
# optimizer.lr (2e-5); pass one to try a different fine-tuning LR without
# editing the yaml.

CHECKPOINT=${1:?Usage: sbatch submit_rexgroundingct_finetune.sh <checkpoint> [override_lr]}
LR=${2:-}

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

EXTRA_ARGS=()
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
  train.py --config configs/rexgroundingct_finetune.yaml \
    --resume "$CHECKPOINT" \
    --resume-partial \
    "${EXTRA_ARGS[@]}" \
    --override model.text_encoder_name=$MODEL_DIR
