#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_l40s_4gpu_10pct_lr3e-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Data-scale ablation: same from-scratch VoxTell recipe as
# submit_voxtell_scratch_l40s_4gpu.sh (randomly initialized encoder/decoder,
# --from-scratch, resampled image/mask grid, no warmup, default 3-channel
# multi-window stack, --random-crop-fraction default 0.33), restricted to
# official_splits/all_data_train_10pct.json (seeded nested 10% subset of
# all_data_train.json -- see scripts/data_prep/make_train_subsets.py), with
# --batch-size 2 / --lr 3e-5 shared across this ablation's siblings (see
# submit_voxtell_scratch_l40s_4gpu_30pct_lr3e-5.sh and
# submit_voxtell_scratch_l40s_4gpu_30pct_lr3e-5_1ch.sh). --batch-size 2 is
# UNTESTED for this from-scratch model's memory profile on 48GB L40S cards
# (finetune_voxtell.py has no gradient checkpointing) -- smoke test first.
#
# --output-dir is its own base so --resume's auto-discovery can't mix
# checkpoints with a different manifest/lr/channel-count sibling run.
#
# Recommend a quick smoke test first:
#   sbatch submit_voxtell_scratch_l40s_4gpu_10pct_lr3e-5.sh --max-samples 20 --num-epochs 2
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_l40s_4gpu_10pct_lr3e-5.sh [extra finetune_voxtell.py args...]

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
    --train-manifest official_splits/all_data_train_10pct.json \
    --output-dir runs/voxtell_scratch_l40s_4gpu_10pct_lr3e-5 \
    --batch-size 2 \
    --lr 3e-5 \
    --warmup-epochs 0 \
    "$@"
