#!/bin/bash
#SBATCH --job-name=grounder_rexval_and_test_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Scores a checkpoint on the ReXGroundingCT val split (official metric, via
# evaluate_rexgroundingct_val.py) and then runs it over the official (blind)
# test set, saving one native-resolution prediction NIfTI per volume (300
# total) to <run_dir>/official_predictions/ -- for submission to the
# ReXGroundingCT leaderboard. <run_dir> is the checkpoint's own run
# directory; neither step takes an output-dir argument, both derive their
# location from the checkpoint path.
#
# Pass --test-only to skip the val-scoring step and only run test
# predictions -- e.g. to retry just the test step after it failed/crashed
# without redoing an already-successful (and much slower) val eval.
#
# Usage: sbatch submit_predict_rexgroundingct_test.sh [--test-only] <checkpoint>
#   e.g. sbatch submit_predict_rexgroundingct_test.sh runs/h200/my_run/checkpoints/best.pt
#        sbatch submit_predict_rexgroundingct_test.sh --test-only runs/h200/my_run/checkpoints/best.pt

TEST_ONLY=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--test-only" ]; then
    TEST_ONLY=1
  else
    ARGS+=("$arg")
  fi
done

CHECKPOINT=${ARGS[0]:?Usage: sbatch submit_predict_rexgroundingct_test.sh [--test-only] <checkpoint>}
RUN_DIR=$(dirname "$(dirname "$(realpath "$CHECKPOINT")")")

DATA_ROOT=/path/to/data
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

if [ "$TEST_ONLY" -eq 0 ]; then
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_rexgroundingct_val.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --data-root $DATA_ROOT \
    --rexrank-eval-script $DATA_ROOT/ReXGroundingCT/rexrank_eval.py \
    --output-dir "$RUN_DIR/rexgroundingct_val_eval" \
    --override model.text_encoder_name=$MODEL_DIR \
      training.batch_size=1
fi

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/predict_rexgroundingct_test.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --data-root $DATA_ROOT \
    --override model.text_encoder_name=$MODEL_DIR \
      training.batch_size=1
