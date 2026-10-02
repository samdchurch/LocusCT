#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_96_h200_4gpu_30pct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# 96^3-input, deeper-encoder counterpart to submit_voxtell_scratch_h200_4gpu.sh
# (192^3). Same from-scratch VoxTell recipe (randomly initialized, --from-scratch,
# official_splits/all_data_train_30pct.json train manifest, lr=1e-4, no warmup,
# resampled image grid, Qwen3-Embedding-4B text encoder, --random-crop-fraction
# default 0.33) -- only --model-dir/--patch-size/--decoder-layer/
# --num-maskformer-stages differ, pointing at voxtell/voxtell_scratch_96/plans.json
# (n_stages=6, features_per_stage unchanged from voxtell_v1.1, strides[1] flipped
# to [1,1,1] so a 96^3 input reproduces VoxTellModel.DECODER_CONFIGS[1..5] exactly,
# n_blocks_per_stage scaled up for depth -- see that plans.json's _description).
# decoder_layer=4/num_maskformer_stages=5 are actually UNCHANGED from finetune_
# voxtell.py's own defaults for this config, but passed explicitly here for
# self-documentation.
#
# batch_size=3 is carried over UNCHANGED from submit_voxtell_scratch_h200_4gpu.sh
# as a conservative starting point -- NOT confirmed to fit this config's memory
# profile (two full-resolution stages instead of one, offset by needing one
# fewer downsampling level elsewhere). The smoke test's nvidia-smi output is
# the real answer, not a back-of-envelope estimate.
#
# MANDATORY before trusting a long job to this config -- the DECODER_CONFIGS
# shape/channel arithmetic has been traced against VoxTellModel's actual source
# (MIC-DKFZ/VoxTell) but never executed against it, since that package isn't
# vendored in this repo:
#   1. A quick standalone forward-pass shape check (see the plan this script was
#      built from) run via `singularity exec grounder.sif python <script>`.
#   2. Then this smoke test:
#        sbatch submit_voxtell_scratch_96_h200_4gpu.sh --max-samples 20 --num-epochs 2
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_96_h200_4gpu.sh [extra finetune_voxtell.py args...]

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
    --train-manifest official_splits/all_data_train_30pct.json \
    --output-dir runs/voxtell_scratch_96_h200_4gpu_30pct \
    --batch-size 3 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    "$@"
