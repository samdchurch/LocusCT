#!/bin/bash
#SBATCH --job-name=grounder_voxtell_merlin_h200_scratch_grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# For our --from-scratch VoxTell checkpoints (e.g. voxtell_scratch_*_lr3e-5)
# -- these train on the resampled 1.5x1.5x3.0mm/352x352x180 grid (see
# resample_and_crop.py), NOT native resolution, and sliding_window_predict
# does no resampling of its own (see finetune_voxtell.py's module docstring)
# -- so this points --image-dir at the existing merlin_data_resampled tree
# (already on that same grid -- see submit_predict_merlin.sh/submit_predict_
# finetuned_voxtell_merlin.sh's own long-standing use of it), NOT the new
# native ICLR2027_merlin_images tree or a freshly-resampled copy of it.
# predict_finetuned_voxtell_merlin.py SKIPs (with a warning) any study whose
# volume is missing under --image-dir, it doesn't fail the whole job -- so if
# this run's exam IDs turn out not to all be covered by merlin_data_resampled,
# resample_merlin.py's --input-root/--output-root/--target-spacing/
# --target-shape overrides (see submit_resample_merlin_voxtell_scratch_grid.sh)
# can produce the missing ones from ICLR2027_merlin_images instead.
#
# Do NOT point --image-dir at ICLR2027_merlin_images directly (the native,
# un-resampled tree) for one of these checkpoints -- that would be a real
# spacing mismatch, not just a shape difference, and would silently produce
# invalid predictions rather than erroring.
#
# Otherwise the same ICLR2027_official_merlin_samples/*.json category-findings
# inference + per-category visualize_merlin_predictions.py loop as
# submit_predict_finetuned_voxtell_merlin.sh. --max-per-category 0 disables
# the script's own default 50-per-category cap, so EVERY (exam, phrase) pair
# in the official sample set runs.
#
# Single H200 (141GB) rather than that script's L40S, and --mem/--time bumped
# (128G/8h vs 64G/4h) -- UNTESTED guesses. Smoke test first, e.g. via
# --max-per-category 1 in extra args (--category-findings-dir always
# overrides --n/--seed sampling per predict_finetuned_voxtell_merlin.py's own
# docs, so --n alone won't limit this run).
#
# --checkpoint MUST be a --from-scratch run's checkpoint (and pass
# --multi-window here too, unless that run used --no-multi-window -- see
# finetune_voxtell.py's --from-scratch defaults). For a checkpoint trained/
# fine-tuned at NATIVE resolution instead (the published pretrained VoxTell,
# or finetune_voxtell.py WITHOUT --from-scratch), use
# submit_predict_finetuned_voxtell_merlin.sh with --image-dir pointed at
# ICLR2027_merlin_images directly instead -- that pairing needs no resampling
# step, this one does.
#
# --text-encoder must match whatever was used for the fine-tuning run
# (finetune_voxtell.py's own default is Qwen3-Embedding-4B, not Grounder's
# 8B).
#
# Any extra arguments are passed straight through to
# predict_finetuned_voxtell_merlin.py (e.g. --tile-overlap, --multi-window,
# --raw-image-dir, --max-per-category).
#
# Usage: sbatch submit_predict_finetuned_voxtell_merlin_h200_scratch_grid.sh <checkpoint> [output_dir] [extra args...]
#   e.g. sbatch submit_predict_finetuned_voxtell_merlin_h200_scratch_grid.sh \
#            runs/voxtell_scratch_h200_4gpu_full_lr3e-5/checkpoints/best.pt \
#            outputs/eval/voxtell_merlin_scratch_grid --multi-window

CHECKPOINT=${1:?Usage: sbatch submit_predict_finetuned_voxtell_merlin_h200_scratch_grid.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/voxtell_merlin_scratch_grid}
shift $(( $# >= 2 ? 2 : $# ))

DATA_ROOT=/path/to/data
MERLIN_IMAGE_DIR=$DATA_ROOT/public_datasets/merlinabdominalctdataset/merlin_data_resampled
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface
CATEGORY_DIR=ICLR2027_official_merlin_samples  # relative to /workspace ($REPO, bound below + --pwd)

mkdir -p $REPO/logs $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/predict_finetuned_voxtell_merlin.py \
    --checkpoint "$CHECKPOINT" \
    --image-dir "$MERLIN_IMAGE_DIR" \
    --text-encoder $TEXT_MODEL_DIR \
    --output-dir "$OUTPUT_DIR" \
    --category-findings-dir "$CATEGORY_DIR" \
    --max-per-category 0 \
    "$@"

# predict_finetuned_voxtell_merlin.py wrote one predictions.json per matched category
# under $OUTPUT_DIR/<category>/ -- visualize each independently into its own viz/
# subfolder (same convention as submit_predict_merlin.sh).
for category_dir in "$OUTPUT_DIR"/*/; do
  predictions_file="$category_dir/predictions.json"
  if [ -f "$predictions_file" ]; then
    singularity run \
      --pwd /workspace \
      --bind $REPO:/workspace \
      $SIF \
      scripts/visualization/visualize_merlin_predictions.py \
        --predictions "$predictions_file" \
        --n 0 \
        --output-dir "$category_dir/viz"
  fi
done
