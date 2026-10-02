#!/bin/bash
#SBATCH --job-name=grounder_voxtell_scratch_96_xl_h200_4gpu_30pct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# "Much bigger" counterpart to submit_voxtell_scratch_96_h200_4gpu.sh -- same
# 96^3 patch/stage schedule, but voxtell/voxtell_scratch_96_xl/plans.json widens
# features_per_stage ([32,64,128,256,320,320] -> [64,128,256,512,512,512]) and
# scales n_blocks_per_stage further ([1,2,6,8,10,10] -> [2,4,10,14,18,18]). This
# means VoxTellModel.DECODER_CONFIGS (the external package's hardcoded table)
# no longer matches this architecture's real per-stage (channels, shape) --
# voxtell_scratch_96_xl/decoder_configs.json is a full replacement table,
# monkey-patched onto VoxTellModel by build_voxtell_model (see its own comment
# for the mechanism). decoder_layer=4/num_maskformer_stages=5 select into that
# replacement table now, not the original -- values happen to be numerically
# the same as voxtell_scratch_96's, but now mean something different (index 4
# = channels 512 here, not 320).
#
# --text-embedding-dim 4096 + Qwen3-Embedding-8B instead of the project's usual
# 4B (2560-dim): --embedding-cache is deliberately OMITTED below (live text
# encoding) since the existing $DATA_ROOT/voxtell_embeddings_mmap cache was
# built from the 4B encoder and is the wrong dimension for this run -- building
# an 8B-encoded cache is a separate precompute step (see
# submit_precompute_voxtell_embeddings.sh, pointed at Qwen3-Embedding-8B and a
# new --output path) if you want the speed-up later; not required to run this.
#
# batch_size=2 (lower than the 96^3 script's 3) is a rougher-than-usual starting
# guess -- this config is both wider AND deeper AND loads the 8B (not 4B) text
# encoder live, all compounding memory pressure well past anything smoke-tested
# so far in this repo. Treat the smoke test's nvidia-smi output as the real
# answer even more than usual.
#
# MANDATORY before trusting a long job to this config -- more so than
# voxtell_scratch_96, since the DECODER_CONFIGS *replacement* mechanism itself
# (not just specific values within the original table) has never been executed:
#   1. A standalone forward-pass shape check equivalent to voxtell_scratch_96's
#      own (see that script's own comment for the pattern), pointed at this
#      model-dir/--text-embedding-dim 4096 instead.
#   2. Then this smoke test:
#        sbatch submit_voxtell_scratch_96_xl_h200_4gpu.sh --max-samples 20 --num-epochs 2 --ed-val-manifest "" --onc-val-manifest ""
#
# Any extra arguments are passed straight through to finetune_voxtell.py.
#
# Usage: sbatch submit_voxtell_scratch_96_xl_h200_4gpu.sh [extra finetune_voxtell.py args...]

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
    --train-manifest official_splits/all_data_train_30pct.json \
    --output-dir runs/voxtell_scratch_96_xl_h200_4gpu_30pct \
    --batch-size 2 \
    --lr 1e-4 \
    --warmup-epochs 0 \
    "$@"
