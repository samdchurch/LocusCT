#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_l40s_4gpu_30pct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# L40S counterpart to submit_voxtell_scratch_h200_4gpu.sh -- same from-scratch
# VoxTell recipe (randomly initialized, --from-scratch, official_splits/
# all_data_train_30pct.json train manifest, lr=1e-4, no warmup), just sized
# for 4x L40S (48GB VRAM/card, vs. H200's 141GB) instead of 4x H200. Per
# submit_finetune_voxtell_l40s.sh's own note, finetune_voxtell.py has no
# gradient checkpointing, so --batch-size 1 is used here -- an UNTESTED
# conservative guess for the from-scratch model's memory profile. Not
# --exclusive and no --exclude=<node>, matching submit_finetune_voxtell_
# l40s.sh (the H200 scratch script's --exclusive/--exclude were H200-node-
# specific, not part of the scratch-training recipe itself).
#
# Recommend a quick smoke test first:
#   sbatch submit_voxtell_scratch_l40s_4gpu.sh --max-samples 20 --num-epochs 2
#
# finetune_voxtell.py's --random-crop-fraction defaults to 0.33 -- 33% of
# training patches are a truly random 192^3 crop (may be empty or clip the
# mask) instead of foreground-guaranteed, so the model sees the kind of
# tiles real sliding-window eval hands it (see finetune_voxtell.py's module
# docstring's "Random crops" section). Pass --random-crop-fraction 0.0 via
# extra args to restore the old 100%-foreground-guaranteed behavior.
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_l40s_4gpu.sh [extra finetune_voxtell.py args...]

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
    --train-manifest official_splits/all_data_train_30pct.json \
    --output-dir runs/voxtell_scratch_l40s_4gpu_30pct \
    --batch-size 1 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    "$@"
