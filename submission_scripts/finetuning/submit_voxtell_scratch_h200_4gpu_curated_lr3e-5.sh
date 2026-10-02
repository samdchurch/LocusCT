#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_h200_4gpu_curated_lr3e-5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same recipe as submit_voxtell_scratch_h200_4gpu_full.sh (from-scratch VoxTell,
# no-warmup/4x H200 DDP, resampled image grid, Qwen3-Embedding-4B text encoder,
# batch_size=3, num_epochs/early_stop_patience left at finetune_voxtell.py's own
# defaults, --random-crop-fraction default 0.33), with --lr 3e-5 instead of that
# script's 1e-4 (matching the earlier ad hoc voxtell_scratch_h200_4gpu_full_lr3e-5
# run), except the training manifest is restricted to the curated ED+ONC train
# sets (official_splits/curated_ed_train_data.json + curated_onc_train_data.json)
# instead of the full all_data_train.json. Those two are merged at the start of
# this job into official_splits/curated_ed_onc_train_data.json via
# scripts/data_prep/merge_manifests.py -- the same tool that already produced
# curated_ed_onc_val_data.json from its ED/ONC val counterparts -- and
# regenerated on every run so it can't go stale if the curated source files change.
#
# --output-dir is its own base (runs/voxtell_scratch_h200_4gpu_curated_lr3e-5) so
# --resume's auto-discovery can't mix checkpoints from a different manifest/lr.
#
# Runs on 2x H200 instead of the original 4x -- --gres and --nproc_per_node
# are both 2. --batch-size is bumped from 3 to 6 (per-GPU) so the effective/
# global batch size (12) matches what this run originally trained with on 4
# GPUs; --batch-size 6 is itself unconfirmed to fit H200's 141GB at this patch
# size (the original batch_size=3 was already only a guess). --job-name/
# --output-dir are kept unchanged to continue the same run/checkpoints.
#
# This script does NOT hardcode --resume -- pass the checkpoint to continue
# from explicitly, since finetune_voxtell.py has no auto-resume and a
# hardcoded path here would go stale as training progresses:
#   sbatch submit_voxtell_scratch_h200_4gpu_curated_lr3e-5.sh \
#     --resume $(ls -v runs/voxtell_scratch_h200_4gpu_curated_lr3e-5/checkpoints/epoch_*.pt | tail -1)
#
# Recommend a quick smoke test first if not resuming an already-validated run:
#   sbatch submit_voxtell_scratch_h200_4gpu_curated_lr3e-5.sh --max-samples 20 --num-epochs 2
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_h200_4gpu_curated_lr3e-5.sh [extra finetune_voxtell.py args...]

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT, same as submit_h200_4gpu.sh
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B  # must be the 4B model -- see note above
EMBEDDING_CACHE=$DATA_ROOT/voxtell_embeddings_mmap  # from submit_precompute_voxtell_embeddings.sh; pass --embedding-cache "" via extra args to fall back to live encoding
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface
TRAIN_MANIFEST=official_splits/curated_ed_onc_train_data.json

mkdir -p $REPO/logs $HF_CACHE

singularity exec \
  --pwd /workspace \
  --bind $REPO:/workspace \
  $SIF \
  python scripts/data_prep/merge_manifests.py \
    --inputs official_splits/curated_ed_train_data.json official_splits/curated_onc_train_data.json \
    --output $TRAIN_MANIFEST

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
  -m torch.distributed.run --nproc_per_node=2 --master_port=$MASTER_PORT \
  finetune_voxtell.py \
    --from-scratch \
    --text-encoder $TEXT_MODEL_DIR \
    --embedding-cache $EMBEDDING_CACHE \
    --train-manifest $TRAIN_MANIFEST \
    --output-dir runs/voxtell_scratch_h200_4gpu_curated_lr3e-5 \
    --batch-size 6 \
    --lr 3e-5 \
    --warmup-epochs 0 \
    "$@"
