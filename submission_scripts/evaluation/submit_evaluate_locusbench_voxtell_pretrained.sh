#!/bin/bash
#SBATCH --job-name=grounder_locusbench_voxtell_pretrained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the ORIGINAL, published VoxTell checkpoint (no fine-tuning) on
# LocusBench-Onc, then LocusBench-ED, which replace the old official ONC/ED
# test sets as the official protocol (submit_evaluate_voxtell_pretrained.sh
# still works against the old sets, just isn't "official" anymore). See that
# script's own comments for why: real voxtell.inference.predictor.
# VoxTellPredictor end-to-end (no resampling needed -- native resolution),
# and the harmless startup delay from its precomputed-embedding-bank download
# attempt failing on this no-internet cluster.
#
# IMAGE_DIR/MASK_DIR both point at each LocusBench split's own raw root
# (unresampled) -- its manifest's "image"/"mask" fields are already prefixed
# with "images/"/"masks/", so one directory serves both.
#
# Usage: sbatch submit_evaluate_locusbench_voxtell_pretrained.sh [output_dir]
#   e.g. sbatch submit_evaluate_locusbench_voxtell_pretrained.sh outputs/eval/locusbench_voxtell_pretrained
#   -> writes outputs/eval/locusbench_voxtell_pretrained/onc/results.json
#      and    outputs/eval/locusbench_voxtell_pretrained/ed/results.json

OUTPUT_DIR=${1:-outputs/eval/locusbench_voxtell_pretrained}

DATA_ROOT=/path/to/data
ONC_MANIFEST=$DATA_ROOT/LocusBench/LocusBench-Onc/LocusBench-Onc.json
ED_MANIFEST=$DATA_ROOT/LocusBench/LocusBench-ED/LocusBench-ED.json
ONC_DIR=$DATA_ROOT/LocusBench/LocusBench-Onc
ED_DIR=$DATA_ROOT/LocusBench/LocusBench-ED
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B
REPO=$HOME/grounder
MODEL_DIR=$REPO/voxtell/voxtell_v1.1  # original published checkpoint, staged in-repo (untracked)
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR/onc $REPO/$OUTPUT_DIR/ed $HF_CACHE

echo "=== LocusBench-Onc ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_voxtell_onc.py \
    --manifest $ONC_MANIFEST \
    --image-dir $ONC_DIR \
    --mask-dir $ONC_DIR \
    --model-dir $MODEL_DIR \
    --text-encoding-model $TEXT_MODEL_DIR \
    --output "$OUTPUT_DIR/onc/results.json"

echo "=== LocusBench-ED ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_voxtell_ed.py \
    --manifest $ED_MANIFEST \
    --image-dir $ED_DIR \
    --mask-dir $ED_DIR \
    --model-dir $MODEL_DIR \
    --text-encoding-model $TEXT_MODEL_DIR \
    --output "$OUTPUT_DIR/ed/results.json"
