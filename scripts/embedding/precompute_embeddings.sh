#!/bin/bash
#SBATCH --job-name=grounder_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT; also where the cache is written (home dir quota is too small for raw hidden-state caches)
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
  precompute_embeddings.py --config configs/default.yaml \
    --output $DATA_ROOT/embeddings_mmap \
    --override model.text_encoder_name=$MODEL_DIR \
      data.train_manifest='[official_splits/all_data_train.json,official_splits/all_train_val_data.json,official_splits/curated_ed_train_data.json,official_splits/curated_ed_train_val_data.json,official_splits/curated_onc_train_data.json,official_splits/curated_onc_train_val_data.json,official_splits/ReXGroundingCT_train.json]' \
      data.val_manifest='[official_splits/curated_ed_val_data.json,official_splits/curated_onc_val_data.json,official_splits/ReXGroundingCT_val.json]' \
      data.test_manifest='[official_splits/all_test_data.json]'
      # ReXGroundingCT_test.json excluded: challenge withholds its masks (mask=null),
      # which the mask-path-keyed cache can't index.
