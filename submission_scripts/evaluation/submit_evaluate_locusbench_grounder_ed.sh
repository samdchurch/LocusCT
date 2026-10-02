#!/bin/bash
#SBATCH --job-name=grounder_locusbench_ed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Scores a Grounder checkpoint on LocusBench-ED, which replaces
# official_splits/ed_official_test_data.json as the official ED protocol
# (submit_ed_official_test_eval.sh still works against the old set, just
# isn't "official" anymore).
#
# IMAGE_DIR/MASK_DIR point at scripts/data_prep/resample_locusbench.py's
# output (same 1.5x1.5x3.0mm/352x352x180 grid the model trains on) -- run
# that script first. RAW_IMAGE_DIR points at LocusBench's own raw images,
# used only by --save_masks to resample predictions back onto the original
# grid.
#
# Usage: sbatch submit_evaluate_locusbench_grounder_ed.sh <checkpoint> [output_dir]
#   e.g. sbatch submit_evaluate_locusbench_grounder_ed.sh runs/h200/checkpoints/best.pt locusbench_grounder_ed

CHECKPOINT=${1:?Usage: sbatch submit_evaluate_locusbench_grounder_ed.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/locusbench_grounder_ed}

DATA_ROOT=/path/to/data
MANIFEST=$DATA_ROOT/LocusBench/LocusBench-ED/LocusBench-ED.json
IMAGE_DIR=$DATA_ROOT/LocusBench-ED_resampled
MASK_DIR=$DATA_ROOT/LocusBench-ED_resampled
RAW_IMAGE_DIR=$DATA_ROOT/LocusBench/LocusBench-ED/images
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
  scripts/evaluation/evaluate_ed_official_test.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --manifest $MANIFEST \
    --image-dir $IMAGE_DIR \
    --mask-dir $MASK_DIR \
    --output "$OUTPUT_DIR/results.json" \
    --save_masks \
    --raw-image-dir $RAW_IMAGE_DIR \
    --override model.text_encoder_name=$MODEL_DIR \
      data.embedding_cache=$DATA_ROOT/locusbench_ed_embeddings_mmap \
      training.batch_size=1
