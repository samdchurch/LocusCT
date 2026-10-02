#!/bin/bash
#SBATCH --job-name=grounder_locusbench_ed_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Precomputes a small, dedicated text-embedding cache for the LocusBench-ED
# official test set (LocusBench/LocusBench-ED/LocusBench-ED.json), which
# replaces official_splits/ed_official_test_data.json as the official ED
# protocol. Needed because checkpoints trained with data.embedding_cache set
# never load a live TextEncoder, so scripts/evaluation/evaluate_ed_official_test.py
# can't tokenize these previously-unseen sentences on the fly -- they have to
# go through the same cache mechanism. (Same reasoning as
# submit_precompute_ed_test_embeddings.sh, which this supersedes.)
#
# Caches raw (pre-projection) hidden states -- each cross-attention stage's
# own text_proj lives in and trains with the UNet, not here, so this cache
# needs no coordination with any other cache to stay consistent. Independent
# of scripts/data_prep/resample_locusbench.py (image resampling) -- this can
# run in parallel with or before/after it.
#
# Run once, then point scripts/evaluation/evaluate_ed_official_test.py (via
# submit_evaluate_locusbench_grounder.sh) at the resulting directory with
# --override data.embedding_cache=/path/to/data/locusbench_ed_embeddings_mmap

DATA_ROOT=/path/to/data  # where the cache is written (home dir quota is too small for raw hidden-state caches)
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/embedding/precompute_embeddings.py --config configs/default.yaml \
    --output $DATA_ROOT/locusbench_ed_embeddings_mmap \
    --override model.text_encoder_name=$MODEL_DIR \
      data.train_manifest=null \
      data.val_manifest=null \
      data.test_manifest=$DATA_ROOT/LocusBench/LocusBench-ED/LocusBench-ED.json
