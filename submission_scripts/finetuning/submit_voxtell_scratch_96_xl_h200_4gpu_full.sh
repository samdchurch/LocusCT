#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_96_xl_h200_4gpu_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Full-data counterpart to submit_voxtell_scratch_96_xl_h200_4gpu.sh (the
# 30pct script) -- same widened/deepened architecture (see that script's own
# comment for the DECODER_CONFIGS replacement-table reasoning), but on
# finetune_voxtell.py's own --train-manifest default (official_splits/
# all_data_train.json) instead of all_data_train_30pct.json, the way
# submit_voxtell_scratch_96_h200_4gpu_full.sh is to
# submit_voxtell_scratch_96_h200_4gpu.sh. lr=1e-5 here (not 1e-4) --
# deliberately lower for the full-data run. --output-dir is its own base
# (runs/voxtell_scratch_96_xl_h200_4gpu_full, not
# runs/voxtell_scratch_96_xl_h200_4gpu_30pct) so --resume's auto-discovery
# can't mix checkpoints trained on different data/LR.
#
# --text-embedding-dim 4096 + Qwen3-Embedding-8B instead of the project's usual
# 4B (2560-dim): --embedding-cache is deliberately OMITTED below (live text
# encoding), same reasoning as the 30pct script -- the existing
# $DATA_ROOT/voxtell_embeddings_mmap cache was built from the 4B encoder.
#
# batch_size=2 carried over from the 30pct script's rough starting guess --
# unconfirmed against the full manifest's memory profile.
#
# MANDATORY before trusting a long job to this config, same as the 30pct
# script -- the DECODER_CONFIGS replacement mechanism has only ever run
# against the 30pct manifest, not this one:
#   1. The standalone forward-pass shape check (see the 30pct script's
#      own comment for the pattern), pointed at this model-dir/
#      --text-embedding-dim 4096 instead.
#   2. Then this smoke test:
#        sbatch submit_voxtell_scratch_96_xl_h200_4gpu_full.sh --max-samples 20 --num-epochs 2 --ed-val-manifest "" --onc-val-manifest ""
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_96_xl_h200_4gpu_full.sh [extra finetune_voxtell.py args...]

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT, same as submit_h200_4gpu.sh
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B  # 8B, not this project's usual 4B -- see note above
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
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $SIF \
  -m torch.distributed.run --nproc_per_node=4 --master_port=$MASTER_PORT \
  finetune_voxtell.py \
    --from-scratch \
    --model-dir voxtell/voxtell_scratch_96_xl \
    --patch-size 96 \
    --decoder-layer 4 \
    --num-maskformer-stages 5 \
    --text-embedding-dim 4096 \
    --text-encoder $TEXT_MODEL_DIR \
    --output-dir runs/voxtell_scratch_96_xl_h200_4gpu_full \
    --batch-size 2 \
    --lr 1e-5 \
    --warmup-epochs 0 \
    "$@"
