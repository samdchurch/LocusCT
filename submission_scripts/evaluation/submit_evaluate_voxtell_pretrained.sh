#!/bin/bash
#SBATCH --job-name=grounder_voxtell_eval_pretrained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the ORIGINAL, published VoxTell checkpoint (no fine-tuning) on the official
# ONC test set, then the official ED test set, via scripts/evaluation/
# evaluate_voxtell_onc.py and evaluate_voxtell_ed.py -- both scored through the real
# voxtell.inference.predictor.VoxTellPredictor exactly as its own package intends:
# crop-to-nonzero + Z-score normalization + true overlapping/Gaussian-blended
# sliding-window inference, all handled internally by the pip package itself. Neither
# script does any preprocessing or tiling of its own (unlike evaluate_finetuned_
# voxtell_{ed,onc}.py, which reimplement a simplified, non-overlapping version of
# that for this repo's own fine-tuned checkpoints and are NOT a faithful
# reproduction of VoxTell's intended inference).
#
# Both scripts' own --model-dir/--image-dir/--mask-dir/--text-encoding-model
# defaults are stale (an old /path/to/data/... filesystem convention, and
# evaluate_voxtell_onc.py's own --mask-dir default of .../ALL_LABELS doesn't match
# the .../labels this repo's other ONC eval scripts actually use) -- all explicitly
# overridden below to match this cluster's current layout.
#
# VoxTellPredictor tries to download a precomputed text-embedding bank from Hugging
# Face Hub on init (use_precomputed_embeddings=True, its own default); this cluster
# has no internet access, so that attempt fails and it falls back to live text
# encoding automatically (a graceful, intended fallback in the package's own code,
# not an error) -- expect a brief delay at startup of each of the two runs below
# while that download attempt times out.
#
# Usage: sbatch submit_evaluate_voxtell_pretrained.sh [output_dir]
#   e.g. sbatch submit_evaluate_voxtell_pretrained.sh outputs/eval/voxtell_pretrained
#   -> writes outputs/eval/voxtell_pretrained/onc/results.json
#      and    outputs/eval/voxtell_pretrained/ed/results.json

OUTPUT_DIR=${1:-outputs/eval/voxtell_pretrained}

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
ONC_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels  # matches submit_evaluate_finetuned_voxtell_onc.sh, not evaluate_voxtell_onc.py's own stale ALL_LABELS default
ED_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/ED_TEST_SET
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B
REPO=$HOME/grounder
MODEL_DIR=$REPO/voxtell/voxtell_v1.1  # original published checkpoint, staged in-repo (untracked) -- see finetune_voxtell.py's own default of the same relative location
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR/onc $REPO/$OUTPUT_DIR/ed $HF_CACHE

echo "=== ONC official test set ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_voxtell_onc.py \
    --image-dir $IMAGE_DIR \
    --mask-dir $ONC_MASK_DIR \
    --model-dir $MODEL_DIR \
    --text-encoding-model $TEXT_MODEL_DIR \
    --output "$OUTPUT_DIR/onc/results.json"

echo "=== ED official test set ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_voxtell_ed.py \
    --image-dir $IMAGE_DIR \
    --mask-dir $ED_MASK_DIR \
    --model-dir $MODEL_DIR \
    --text-encoding-model $TEXT_MODEL_DIR \
    --output "$OUTPUT_DIR/ed/results.json"
