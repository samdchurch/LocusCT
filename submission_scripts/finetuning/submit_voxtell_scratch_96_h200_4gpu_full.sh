#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_96_h200_4gpu_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same as submit_voxtell_scratch_96_h200_4gpu.sh, but on the FULL training
# manifest (finetune_voxtell.py's own --train-manifest default, official_splits/
# all_data_train.json) instead of the 30pct subset -- the full-data counterpart
# to that 30pct script, the way submit_voxtell_scratch_h200_4gpu_full.sh is to
# submit_voxtell_scratch_h200_4gpu.sh for the 192^3 config. --output-dir is its
# own base (runs/voxtell_scratch_96_h200_4gpu_full, not
# runs/voxtell_scratch_96_h200_4gpu_30pct) so --resume's auto-discovery can't
# mix checkpoints trained on different data.
#
# See submit_voxtell_scratch_96_h200_4gpu.sh for the rest of this recipe's
# reasoning (--model-dir voxtell/voxtell_scratch_96, --patch-size 96,
# --decoder-layer 4/--num-maskformer-stages 5 unchanged from finetune_voxtell.py's
# own defaults, lr=1e-4/no-warmup/4x H200 DDP, batch_size=3 -- NOT confirmed to
# fit this config's memory profile, --random-crop-fraction default 0.33) --
# unchanged here except for the manifest/output-dir.
#
# MANDATORY before trusting a long job to this config, same as the 30pct
# script -- the DECODER_CONFIGS shape/channel arithmetic has been traced
# against VoxTellModel's actual source but never executed against it:
#   1. The standalone forward-pass shape check (see the plan
#      submit_voxtell_scratch_96_h200_4gpu.sh was built from).
#   2. A smoke test:
#        sbatch submit_voxtell_scratch_96_h200_4gpu_full.sh --max-samples 20 --num-epochs 2 --ed-val-manifest "" --onc-val-manifest ""
#      (the --ed-val-manifest/--onc-val-manifest ""s avoid macro_hit_rate_epoch's
#      own guard erroring on too few validation samples per category at
#      --max-samples 20 -- see finetune_voxtell.py's macro_hit_rate_epoch)
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_96_h200_4gpu_full.sh [extra finetune_voxtell.py args...]

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT, same as submit_h200_4gpu.sh
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B  # must be the 4B model -- see note above
EMBEDDING_CACHE=$DATA_ROOT/voxtell_embeddings_mmap  # from submit_precompute_voxtell_embeddings.sh; text-only, reused unchanged across patch sizes/architectures; pass --embedding-cache "" via extra args to fall back to live encoding
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
    --model-dir voxtell/voxtell_scratch_96 \
    --patch-size 96 \
    --decoder-layer 4 \
    --num-maskformer-stages 5 \
    --text-encoder $TEXT_MODEL_DIR \
    --embedding-cache $EMBEDDING_CACHE \
    --output-dir runs/voxtell_scratch_96_h200_4gpu_full \
    --batch-size 3 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    "$@"
