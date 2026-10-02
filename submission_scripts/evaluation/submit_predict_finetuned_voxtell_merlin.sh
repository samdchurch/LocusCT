#!/bin/bash
#SBATCH --job-name=grounder_voxtell_merlin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# VoxTell counterpart to submit_predict_merlin.sh (our own Grounder model's
# version) -- runs scripts/evaluation/predict_finetuned_voxtell_merlin.py
# directly on ICLR2027_official_merlin_samples/*.json (one file per ED
# category), using each phrase as-is as the referring expression, then runs
# scripts/visualization/visualize_merlin_predictions.py over each category
# folder's own predictions.json. See predict_merlin.py's --category-findings-dir
# docs for why there's deliberately NO cross-reference against
# merlin_sentences.json. --max-per-category 0 disables the script's own default
# 50-per-category cap, so EVERY (exam, phrase) pair in this official sample set
# runs. Visualization is CPU-only but tacked onto this same GPU job rather than
# a second submission, same reasoning as submit_predict_merlin.sh.
#
# --text-encoder must match whatever was used for the fine-tuning run
# (finetune_voxtell.py's own default is Qwen3-Embedding-4B, not Grounder's
# 8B). --multi-window must also be passed here if the checkpoint was
# fine-tuned with --multi-window.
#
# Any extra arguments are passed straight through to
# predict_finetuned_voxtell_merlin.py (e.g. --tile-overlap, --raw-image-dir).
#
# Usage: sbatch submit_predict_finetuned_voxtell_merlin.sh <checkpoint> [output_dir] [extra args...]
#   e.g. sbatch submit_predict_finetuned_voxtell_merlin.sh \
#            runs/voxtell_finetune/checkpoints/best.pt outputs/eval/voxtell_merlin_predictions

CHECKPOINT=${1:?Usage: sbatch submit_predict_finetuned_voxtell_merlin.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/voxtell_merlin_predictions}
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
