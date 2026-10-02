#!/bin/bash
#SBATCH --job-name=grounder_merlin_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Precomputes a text-embedding cache for merlin_sentences.json's "present" atomic
# findings. Needed because checkpoints trained with data.embedding_cache set (this
# repo's default) never load a live TextEncoder, so scripts/evaluation/predict_merlin.py can't
# tokenize Merlin's previously-unseen sentences on the fly with those checkpoints
# -- they have to go through this same cache mechanism instead. (Same reasoning as
# submit_precompute_onc_test_embeddings.sh / submit_precompute_ed_test_embeddings.sh.)
#
# Run once, then point scripts/evaluation/predict_merlin.py (via submit_predict_merlin.sh's optional
# 4th argument) at the resulting directory.

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
  scripts/embedding/precompute_merlin_embeddings.py --config configs/default.yaml \
    --image-dir $DATA_ROOT/public_datasets/merlinabdominalctdataset/merlin_data_resampled \
    --output $DATA_ROOT/merlin_embeddings_mmap \
    --override model.text_encoder_name=$MODEL_DIR
