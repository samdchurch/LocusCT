#!/bin/bash
#SBATCH --job-name=grounder_merlin_192
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same as submit_predict_merlin.sh, but against merlin_data_resampled_192 (the
# 192^3 @ 2.0x2.0x3.0mm grid from scripts/data_prep/resample_merlin.py) instead
# of the default 352x352x180 grid, for checkpoints trained on that grid (e.g.
# submit_mw_h200.sh: multi_window=true, gated_cross_attention fusion,
# unet_base_channels=16). These model.*/data.multi_window overrides must match
# whatever checkpoint you're evaluating -- state_dict loading will fail on a
# shape mismatch otherwise. Adjust them if evaluating a different 192-grid
# checkpoint (e.g. the paused group_by_image run's).
#
# Runs scripts/evaluation/predict_merlin.py over a random --n sample of merlin_sentences.json's "present"
# atomic findings, then scripts/visualization/visualize_merlin_predictions.py over all of the resulting
# predictions -- scripts/evaluation/predict_merlin.py only runs inference on the sample that's actually
# going to be looked at (see its own --n/--seed), so N here controls both how many
# get predicted AND how many get visualized, rather than predicting everything and
# visualizing a subset. Visualization is CPU-only but tacked onto this same GPU job
# rather than a second submission, since the marginal GPU idle time is negligible
# next to a second job's queue wait.
#
# Uses the embedding cache from submit_precompute_merlin_embeddings.sh automatically
# if it exists at its default output path (much faster: skips the 8B-parameter frozen
# text encoder entirely) -- falls back to live text encoding otherwise. That cache is
# text-only (keyed by study/finding, not image resolution), so the same one built
# against merlin_data_resampled works here too -- pass a 4th argument to point at a
# different cache directory, or "none" to force live encoding.
#
# Usage: sbatch submit_predict_merlin_192.sh <checkpoint> [output_dir] [n] [embedding_cache]
#   e.g. sbatch submit_predict_merlin_192.sh runs/h200_restart_full/<run_name>/checkpoints/best.pt

CHECKPOINT=${1:?Usage: sbatch submit_predict_merlin_192.sh <checkpoint> [output_dir] [n] [embedding_cache]}
OUTPUT_DIR=${2:-outputs/eval/merlin_predictions_192}
N=${3:-20}

DATA_ROOT=/path/to/data
MERLIN_IMAGE_DIR=$DATA_ROOT/public_datasets/merlinabdominalctdataset/merlin_data_resampled_192
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

EMBEDDING_CACHE=${4:-$DATA_ROOT/merlin_embeddings_mmap}

mkdir -p $REPO/logs $HF_CACHE

EXTRA_ARGS=()
if [ "$EMBEDDING_CACHE" != "none" ] && [ -d "$EMBEDDING_CACHE" ]; then
  echo "Using embedding cache: $EMBEDDING_CACHE"
  EXTRA_ARGS+=(--embedding-cache "$EMBEDDING_CACHE")
else
  echo "No embedding cache at $EMBEDDING_CACHE -- falling back to live text encoding (run submit_precompute_merlin_embeddings.sh first to build one)"
fi

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
    --n "$N" \
    "${EXTRA_ARGS[@]}" \
    --override model.text_encoder_name=$MODEL_DIR \
      data.spatial_size=[192,192,192] \
      data.multi_window=true \
      model.fusion_type=gated_cross_attention \
      model.unet_base_channels=16 \
      training.batch_size=1

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  $SIF \
  scripts/visualization/visualize_merlin_predictions.py \
    --predictions "$OUTPUT_DIR/predictions.json" \
    --n 0 \
    --output-dir "$OUTPUT_DIR/viz"
