#!/bin/bash
#SBATCH --job-name=grounder_biomedparse_eval_pretrained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the ORIGINAL, published BiomedParse-v2 checkpoint (no fine-tuning) on the
# official ONC test set, then the official ED test set, via scripts/evaluation/
# evaluate_biomedparse_onc.py and evaluate_biomedparse_ed.py -- both loaded and run
# exactly as biomedparse-v2/segmentation_example.py intends (hydra-instantiated from
# configs/model/biomedparse_3D.yaml, weights via model.load_pretrained(), forward
# pass via model(..., mode="eval")), with CT windowing/rescaling to [0,255] applied
# first per the BiomedParse-v2 README's own "Recommended Preprocessing" section.
# Each finding gets its own single-prompt forward pass rather than batching a scan's
# findings together, so overlapping findings don't distort each other's Dice (see
# both scripts' own docstrings) -- this makes the run noticeably slower than
# VoxTell's per-scan sliding-window pass, hence the longer time limit above.
#
# Prerequisite: this needs a SEPARATE Singularity image from grounder.sif.
# BiomedParse pulls in a much heavier, incompatible dependency stack (detectron2
# built from git HEAD with compiled CUDA kernels, hydra-core, lightning, open-clip,
# nnunetv2 for NibabelIOWithReorient) -- see biomedparse-v2/Dockerfile. Build it the
# same way grounder.sif was built (Dockerfile -> registry -> `singularity pull`,
# since `singularity build --fakeroot`/`--remote` don't work on this cluster):
#   docker build -t <registry>/biomedparse:latest -f biomedparse-v2/Dockerfile biomedparse-v2/
#   docker push <registry>/biomedparse:latest
#   singularity pull biomedparse.sif docker://<registry>/biomedparse:latest
# and place the resulting biomedparse.sif at $REPO/biomedparse.sif (or point SIF
# below at wherever it ends up).
#
# Both scripts' own --image-dir/--mask-dir defaults are stale (an old
# /path/to/data/... filesystem convention), and --biomedparse-repo/--checkpoint
# default to a "biomedparse-v2" subfolder next to the scripts themselves -- all
# explicitly overridden below to match this cluster's current layout.
#
# biomedparse-v2/ lives under $DATA_ROOT/models/ (not the grounder repo root --
# it was moved out since it bundles the BiomedParse source, our wrapper scripts,
# AND the model checkpoint together, same shared-models convention as
# models/Qwen3-Embedding-8B/), so BIOMEDPARSE_REPO/CHECKPOINT below are absolute
# paths under $DATA_ROOT rather than paths relative to /workspace -- already
# visible in the container via the existing --bind $DATA_ROOT:$DATA_ROOT below,
# no separate bind needed.
#
# CLIP_TOKENIZER_DIR (--clip-tokenizer-dir): BiomedParse-v2's own hydra config
# (configs/model/sem_seg_head/predictor/language_encoder/seem_language_encoder.yaml)
# hardcodes 'openai/clip-vit-base-patch32' as an HF Hub ID for its internal CLIP
# tokenizer (vocab/merges only -- LOAD_PRETRAINED: false means the actual encoder
# weights come from --checkpoint, not from HF). On this cluster's no-internet-
# egress compute nodes that fetch hard-fails (ConnectionResetError -> OSError)
# rather than falling back to a cache, unlike VoxTellPredictor's embedding-bank
# download attempt elsewhere. Fix: from a machine WITH internet access, run
#   python -c "from transformers import AutoTokenizer; \
#     AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32', use_fast=False).save_pretrained('clip-vit-base-patch32')"
# then copy the resulting clip-vit-base-patch32/ directory to
# $DATA_ROOT/models/clip-vit-base-patch32 (or point CLIP_TOKENIZER_DIR elsewhere).
#
# Usage: sbatch submit_evaluate_biomedparse_pretrained.sh [output_dir]
#   e.g. sbatch submit_evaluate_biomedparse_pretrained.sh outputs/eval/biomedparse_pretrained
#   -> writes outputs/eval/biomedparse_pretrained/onc/results.json
#      and    outputs/eval/biomedparse_pretrained/ed/results.json

OUTPUT_DIR=${1:-outputs/eval/biomedparse_pretrained}

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
ONC_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels
ED_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/ED_TEST_SET
REPO=$HOME/grounder
BIOMEDPARSE_REPO=$DATA_ROOT/models/biomedparse-v2/BiomedParse
CHECKPOINT=$DATA_ROOT/models/biomedparse-v2/model/biomedparse_v2.ckpt
CLIP_TOKENIZER_DIR=$DATA_ROOT/models/clip-vit-base-patch32  # see note above -- must be pre-staged manually
SIF=$REPO/biomedparse.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR/onc $REPO/$OUTPUT_DIR/ed $HF_CACHE

echo "=== ONC official test set ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  python scripts/evaluation/evaluate_biomedparse_onc.py \
    --image-dir $IMAGE_DIR \
    --mask-dir $ONC_MASK_DIR \
    --biomedparse-repo $BIOMEDPARSE_REPO \
    --checkpoint $CHECKPOINT \
    --clip-tokenizer-dir $CLIP_TOKENIZER_DIR \
    --output "$OUTPUT_DIR/onc/results.json"

echo "=== ED official test set ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  python scripts/evaluation/evaluate_biomedparse_ed.py \
    --image-dir $IMAGE_DIR \
    --mask-dir $ED_MASK_DIR \
    --biomedparse-repo $BIOMEDPARSE_REPO \
    --checkpoint $CHECKPOINT \
    --clip-tokenizer-dir $CLIP_TOKENIZER_DIR \
    --output "$OUTPUT_DIR/ed/results.json"
