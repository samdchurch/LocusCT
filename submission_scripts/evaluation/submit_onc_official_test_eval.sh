#!/bin/bash
#SBATCH --job-name=grounder_onc_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_onc_official_test_eval.sh <checkpoint> [output_dir]
#   e.g. sbatch submit_onc_official_test_eval.sh runs/h200/checkpoints/best.pt onc_official_test_eval

CHECKPOINT=${1:?Usage: sbatch submit_onc_official_test_eval.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/onc_official_test}

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti_resampled
MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels_resampled
RAW_IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_onc_official_test.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --image-dir $IMAGE_DIR \
    --mask-dir $MASK_DIR \
    --output "$OUTPUT_DIR/results.json" \
    --save_masks \
    --raw-image-dir $RAW_IMAGE_DIR \
    --override model.text_encoder_name=$MODEL_DIR \
      data.embedding_cache=$DATA_ROOT/onc_official_test_embeddings_mmap \
      training.batch_size=1
