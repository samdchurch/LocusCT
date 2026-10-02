#!/bin/bash
#SBATCH --job-name=grounder_locusbench_voxtell_finetuned_onc_1ch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Single-channel counterpart to submit_evaluate_locusbench_voxtell_finetuned_onc.sh:
# same resampled-grid LocusBench-Onc eval, but omits that script's hardcoded
# --multi-window (evaluate_finetuned_voxtell_onc.py's own --multi-window
# already defaults to False/single-channel, and has no --no-multi-window
# negation flag to cancel it back out via extra args -- so a checkpoint
# trained WITHOUT --multi-window, e.g. voxtell_scratch_l40s_4gpu_30pct_lr3e-5_1ch,
# needs its own script rather than an override). Matches
# submit_evaluate_finetuned_voxtell_onc_1ch.sh's reasoning, repointed at
# LocusBench-Onc (which replaces official_splits/onc_official_test_data.json
# as the official oncology protocol).
#
# Usage: sbatch submit_evaluate_locusbench_voxtell_finetuned_onc_1ch.sh <checkpoint> [output_dir] [extra args...]
#
# Any extra arguments beyond checkpoint/output_dir are passed straight
# through to evaluate_finetuned_voxtell_onc.py (e.g. --tile-overlap).
#
# --text-encoder must match whatever was used for the fine-tuning run
# (finetune_voxtell.py's own default is Qwen3-Embedding-4B, not Grounder's 8B).
#
# IMAGE_DIR/MASK_DIR point at scripts/data_prep/resample_locusbench.py's
# output (same grid the --from-scratch VoxTell checkpoint trains on) -- run
# that script first. Channel count doesn't change which grid a --from-scratch
# checkpoint expects, only --multi-window does.

CHECKPOINT=${1:?Usage: sbatch submit_evaluate_locusbench_voxtell_finetuned_onc_1ch.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/locusbench_voxtell_finetuned_onc_1ch}
shift $(( $# >= 2 ? 2 : $# ))

DATA_ROOT=/path/to/data
MANIFEST=$DATA_ROOT/LocusBench/LocusBench-Onc/LocusBench-Onc.json
IMAGE_DIR=$DATA_ROOT/LocusBench-Onc_resampled
MASK_DIR=$DATA_ROOT/LocusBench-Onc_resampled
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
  scripts/evaluation/evaluate_finetuned_voxtell_onc.py \
    --checkpoint "$CHECKPOINT" \
    --manifest $MANIFEST \
    --image-dir $IMAGE_DIR \
    --mask-dir $MASK_DIR \
    --text-encoder $TEXT_MODEL_DIR \
    --output "$OUTPUT_DIR/results.json" \
    "$@"
