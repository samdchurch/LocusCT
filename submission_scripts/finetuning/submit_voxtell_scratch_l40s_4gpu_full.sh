#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_l40s_4gpu_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same as submit_voxtell_scratch_l40s_4gpu.sh, but on the FULL training
# manifest (finetune_voxtell.py's own --train-manifest default,
# official_splits/all_data_train.json) instead of the 30pct subset -- the
# L40S counterpart to submit_voxtell_scratch_h200_4gpu_full.sh, the way
# submit_voxtell_scratch_l40s_4gpu.sh is the L40S counterpart to
# submit_voxtell_scratch_h200_4gpu.sh. --output-dir is its own base
# (runs/voxtell_scratch_l40s_4gpu_full, not runs/voxtell_scratch_l40s_4gpu_30pct)
# so --resume's auto-discovery can't mix checkpoints trained on different data.
#
# See submit_voxtell_scratch_l40s_4gpu.sh for the rest of this recipe's
# reasoning (lr=1e-4/no-warmup/4x L40S DDP, resampled image grid, Qwen3-
# Embedding-4B text encoder, batch_size=1 -- an UNTESTED conservative guess,
# no gradient checkpointing in finetune_voxtell.py so it can't be lowered
# further if it OOMs, --random-crop-fraction default 0.33) -- unchanged here
# except for the manifest/output-dir. Not --exclusive, same as submit_voxtell_
# scratch_l40s_4gpu.sh -- this job may share a node with others.
#
# Recommend a quick smoke test first, doubly so at full-manifest scale:
#   sbatch submit_voxtell_scratch_l40s_4gpu_full.sh --max-samples 20 --num-epochs 2
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_l40s_4gpu_full.sh [extra finetune_voxtell.py args...]

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT, same as submit_voxtell_scratch_h200_4gpu.sh
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B  # must be the 4B model -- build_voxtell_model hardcodes text_embedding_dim=2560 to match VoxTellModel's own text-fusion decoder
EMBEDDING_CACHE=$DATA_ROOT/voxtell_embeddings_mmap  # from submit_precompute_voxtell_embeddings.sh; pass --embedding-cache "" via extra args to fall back to live encoding
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derived from the job ID, not hardcoded -- avoids EADDRINUSE against another
# torch.distributed.run job sharing the node (this job isn't --exclusive, so
# it may share a node with other jobs).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $SIF \
  -m torch.distributed.run --nproc_per_node=4 --master_port=$MASTER_PORT \
  finetune_voxtell.py \
    --from-scratch \
    --text-encoder $TEXT_MODEL_DIR \
    --embedding-cache $EMBEDDING_CACHE \
    --output-dir runs/voxtell_scratch_l40s_4gpu_full \
    --batch-size 1 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    "$@"
