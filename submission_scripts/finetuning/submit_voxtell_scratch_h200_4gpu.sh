#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_h200_4gpu_30pct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Architecture side-by-side counterpart to submit_h200_4gpu_30pct.sh: same
# SBATCH resources, same DATA_ROOT, same lr=1e-4/no-warmup/4x H200 DDP
# training recipe, same official_splits/all_data_train_30pct.json train
# manifest (all other manifests -- val/ed-val/onc-val -- stay full-size,
# matching submit_h200_4gpu_30pct.sh, which only subsets data.train_manifest)
# -- just VoxTell's architecture (randomly initialized, --from-scratch, see
# finetune_voxtell.py's module docstring) in place of Grounder's own
# from-scratch UNet3D. num_epochs/early_stop_patience left at finetune_
# voxtell.py's own config-matching defaults (100 / 4), same as submit_h200_
# 4gpu_30pct.sh leaves train.py's.
#
# From-scratch VoxTell training is pinned to the 30pct subset (not the full
# manifest, unlike submit_h200_4gpu.sh) -- pass --train-manifest to override.
#
# One difference from submit_h200_4gpu.sh is forced by VoxTell's
# architecture, not chosen for this comparison: text encoder is
# Qwen3-Embedding-4B, not Grounder's default 8B -- build_voxtell_model
# hardcodes text_embedding_dim=2560 to match VoxTellModel's own text-fusion
# decoder.
#
# --image-dir/--mask-dir: --from-scratch defaults these to configs/
# default.yaml's own resampled nifti_resampled/labels_resampled grid (same
# fair-comparison reasoning as submit_h200_4gpu_30pct.sh's input), not the
# native (pre-resample) resolution pretrained-checkpoint fine-tuning uses --
# see finetune_voxtell.py's "Image resolution" docstring section. VoxTellModel
# still only ever sees a 192^3 patch cropped/tiled out of whichever volume
# gets loaded either way (its positional-encoding buffer is fixed to that
# size) -- this only changes which grid those patches are cropped from.
#
# batch_size=3 is carried over unmodified from submit_h200_4gpu_30pct.sh --
# NOT confirmed to fit VoxTell's own 192^3-patch memory profile (untested).
# Recommend a quick smoke test first:
#   sbatch submit_voxtell_scratch_h200_4gpu.sh --max-samples 20 --num-epochs 2
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
# Usage: sbatch submit_voxtell_scratch_h200_4gpu.sh [extra finetune_voxtell.py args...]

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT, same as submit_h200_4gpu.sh
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B  # must be the 4B model -- see note above
EMBEDDING_CACHE=$DATA_ROOT/voxtell_embeddings_mmap  # from submit_precompute_voxtell_embeddings.sh; pass --embedding-cache "" via extra args to fall back to live encoding
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
    --text-encoder $TEXT_MODEL_DIR \
    --embedding-cache $EMBEDDING_CACHE \
    --train-manifest official_splits/all_data_train_30pct.json \
    --output-dir runs/voxtell_scratch_h200_4gpu_30pct \
    --batch-size 3 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    "$@"
