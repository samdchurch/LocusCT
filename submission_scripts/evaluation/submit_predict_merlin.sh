#!/bin/bash
#SBATCH --job-name=grounder_merlin
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs scripts/evaluation/predict_merlin.py directly on
# ICLR2027_official_merlin_samples/*.json (one file per ED category -- each a
# list of {"exam": study_id, "phrase": finding text}), using each phrase as-is
# as the referring expression -- see predict_merlin.py's --category-findings-dir
# docs for why there's deliberately NO cross-reference against
# merlin_sentences.json (matching phrase text against it was tried and found
# essentially unusable: ~2% exam overlap, and <10% exact text match even within that
# overlap -- a different, differently-derived corpus, not a text-normalization bug).
# --n/--seed random subsampling is overridden entirely; --max-per-category 0 disables
# predict_merlin.py's own default 50-per-category cap, so EVERY (exam, phrase) pair in
# this official sample set runs, not a random subset -- an (exam, phrase) pair
# referenced by more than one category's file is saved into each of those category
# folders. Results are saved under $OUTPUT_DIR/<category>/. Then runs
# scripts/visualization/visualize_merlin_predictions.py over each category folder's own
# predictions.json. Visualization is CPU-only but tacked onto this same GPU job rather
# than a second submission, since the marginal GPU idle time is negligible next to a
# second job's queue wait.
#
# Always uses live text encoding (no --embedding-cache) -- these samples were never
# part of merlin_sentences.json, which precompute_merlin_embeddings.py's cache is
# keyed from, so they can never be in it.
#
# Usage: sbatch submit_predict_merlin.sh <checkpoint> [output_dir]
#   e.g. sbatch submit_predict_merlin.sh runs/h200_restart_full/<run_name>/checkpoints/best.pt

CHECKPOINT=${1:?Usage: sbatch submit_predict_merlin.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/merlin_predictions}

DATA_ROOT=/path/to/data
MERLIN_IMAGE_DIR=$DATA_ROOT/public_datasets/merlinabdominalctdataset/merlin_data_resampled
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface
CATEGORY_DIR=ICLR2027_official_merlin_samples  # relative to /workspace ($REPO, bound below + --pwd) -- consistent with how the python script path itself is referenced, rather than relying on Singularity's default $HOME auto-bind

mkdir -p $REPO/logs $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/predict_merlin.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --image-dir "$MERLIN_IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --category-findings-dir "$CATEGORY_DIR" \
    --max-per-category 0 \
    --override model.text_encoder_name=$MODEL_DIR \
      training.batch_size=1

# predict_merlin.py wrote one predictions.json per matched category under $OUTPUT_DIR/<category>/ --
# visualize each independently into its own viz/ subfolder. (This same loop is also
# available standalone as submit_visualize_merlin_predictions.sh, for re-running
# visualization alone without redoing inference -- e.g. if this step reports "No such
# file or no access" for predicted_masks/ files that DO exist moments later, a known
# write-visibility lag on this repo's network share between separate singularity run
# invocations.)
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
