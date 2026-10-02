#!/bin/bash
#SBATCH --job-name=grounder_locusbench_voxtell_finetuned_ed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Scores a finetuned-VoxTell checkpoint on LocusBench-ED, which replaces
# official_splits/ed_official_test_data.json as the official ED protocol
# (submit_evaluate_finetuned_voxtell_ed.sh still works against the old set,
# just isn't "official" anymore).
#
# Usage: sbatch submit_evaluate_locusbench_voxtell_finetuned_ed.sh <checkpoint> [output_dir] [extra args...]
#
# Any extra arguments beyond checkpoint/output_dir are passed straight
# through to evaluate_finetuned_voxtell_ed.py (e.g. --tile-overlap).
#
# --text-encoder must match whatever was used for the fine-tuning run
# (finetune_voxtell.py's own default is Qwen3-Embedding-4B, not Grounder's
# 8B). --multi-window must also be passed here if the checkpoint was
# fine-tuned with --multi-window.
#
# IMAGE_DIR/MASK_DIR point at scripts/data_prep/resample_locusbench.py's
# output (same grid the --from-scratch VoxTell checkpoint trains on) -- run
# that script first. Switch to LocusBench's raw images/masks instead to
# evaluate a checkpoint fine-tuned from the pretrained VoxTell weights,
# which trains on native resolution.

CHECKPOINT=${1:?Usage: sbatch submit_evaluate_locusbench_voxtell_finetuned_ed.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/locusbench_voxtell_finetuned_ed}
shift $(( $# >= 2 ? 2 : $# ))

DATA_ROOT=/path/to/data
MANIFEST=$DATA_ROOT/LocusBench/LocusBench-ED/LocusBench-ED.json
IMAGE_DIR=$DATA_ROOT/LocusBench-ED_resampled
MASK_DIR=$DATA_ROOT/LocusBench-ED_resampled
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_finetuned_voxtell_ed.py \
    --checkpoint "$CHECKPOINT" \
    --manifest $MANIFEST \
    --image-dir $IMAGE_DIR \
    --mask-dir $MASK_DIR \
    --text-encoder $TEXT_MODEL_DIR \
    --multi-window \
    --output "$OUTPUT_DIR/results.json" \
    "$@"
