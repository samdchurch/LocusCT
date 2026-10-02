#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_l40s_4gpu_30pct_lr3e-5_1ch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Input-channel-count ablation sibling of submit_voxtell_scratch_l40s_4gpu_
# 30pct_lr3e-5.sh: identical recipe (from-scratch VoxTell, official_splits/
# all_data_train_30pct.json, batch_size=2, lr=3e-5, no warmup, resampled
# image/mask grid, --random-crop-fraction default 0.33), except this run
# passes --no-multi-window to force finetune_voxtell.py's original single
# Z-score-normalized channel (input_channels=1) instead of --from-scratch's
# own default 3-channel lung/soft-tissue/bone multi-window stack
# (input_channels=3). --no-multi-window requires finetune_voxtell.py's
# parse_args() to define it (added alongside the pre-existing --multi-window
# flag, both dest="multi_window") -- confirm it's present before submitting.
#
# --output-dir is its own base so --resume's auto-discovery can't mix
# 1-channel checkpoints with the 3-channel 30pct_lr3e-5 sibling run.
#
# Recommend a quick smoke test first, and confirm the startup log prints
# "Building VoxTell model (input_channels=1)" (not 3):
#   sbatch submit_voxtell_scratch_l40s_4gpu_30pct_lr3e-5_1ch.sh --max-samples 20 --num-epochs 2
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_l40s_4gpu_30pct_lr3e-5_1ch.sh [extra finetune_voxtell.py args...]

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
    --no-multi-window \
    --text-encoder $TEXT_MODEL_DIR \
    --embedding-cache $EMBEDDING_CACHE \
    --train-manifest official_splits/all_data_train_30pct.json \
    --output-dir runs/voxtell_scratch_l40s_4gpu_30pct_lr3e-5_1ch \
    --batch-size 2 \
    --lr 3e-5 \
    --warmup-epochs 0 \
    "$@"
