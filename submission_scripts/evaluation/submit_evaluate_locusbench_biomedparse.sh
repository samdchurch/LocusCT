#!/bin/bash
#SBATCH --job-name=grounder_locusbench_biomedparse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the ORIGINAL, published BiomedParse-v2 checkpoint (no fine-tuning) on
# LocusBench-Onc, then LocusBench-ED, which replace the old official ONC/ED
# test sets as the official protocol
# (submit_evaluate_biomedparse_pretrained.sh still works against the old
# sets, just isn't "official" anymore). See that script's own comments for
# prerequisites (separate biomedparse.sif, pre-staged clip-vit-base-patch32
# tokenizer) -- all identical here, none of that changes for LocusBench.
#
# IMAGE_DIR/MASK_DIR both point at each LocusBench split's own raw root
# (unresampled) -- BiomedParse-v2 does its own CT windowing/rescaling to
# [0,255] from raw NIfTI, same as it already does against the old official
# test sets.

OUTPUT_DIR=${1:-outputs/eval/locusbench_biomedparse}

DATA_ROOT=/path/to/data
ONC_MANIFEST=$DATA_ROOT/LocusBench/LocusBench-Onc/LocusBench-Onc.json
ED_MANIFEST=$DATA_ROOT/LocusBench/LocusBench-ED/LocusBench-ED.json
ONC_DIR=$DATA_ROOT/LocusBench/LocusBench-Onc
ED_DIR=$DATA_ROOT/LocusBench/LocusBench-ED
REPO=$HOME/grounder
BIOMEDPARSE_REPO=$DATA_ROOT/models/biomedparse-v2/BiomedParse
CHECKPOINT=$DATA_ROOT/models/biomedparse-v2/model/biomedparse_v2.ckpt
CLIP_TOKENIZER_DIR=$DATA_ROOT/models/clip-vit-base-patch32  # see submit_evaluate_biomedparse_pretrained.sh -- must be pre-staged manually
SIF=$REPO/biomedparse.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR/onc $REPO/$OUTPUT_DIR/ed $HF_CACHE

echo "=== LocusBench-Onc ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  python scripts/evaluation/evaluate_biomedparse_onc.py \
    --manifest $ONC_MANIFEST \
    --image-dir $ONC_DIR \
    --mask-dir $ONC_DIR \
    --biomedparse-repo $BIOMEDPARSE_REPO \
    --checkpoint $CHECKPOINT \
    --clip-tokenizer-dir $CLIP_TOKENIZER_DIR \
    --output "$OUTPUT_DIR/onc/results.json"

echo "=== LocusBench-ED ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  python scripts/evaluation/evaluate_biomedparse_ed.py \
    --manifest $ED_MANIFEST \
    --image-dir $ED_DIR \
    --mask-dir $ED_DIR \
    --biomedparse-repo $BIOMEDPARSE_REPO \
    --checkpoint $CHECKPOINT \
    --clip-tokenizer-dir $CLIP_TOKENIZER_DIR \
    --output "$OUTPUT_DIR/ed/results.json"
