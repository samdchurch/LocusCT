#!/bin/bash
#SBATCH --job-name=grounder_voxtell_rexval_and_test_predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Scores a fine-tuned VoxTell checkpoint (finetune_voxtell.py's model_state_dict
# format) on the ReXGroundingCT val split (evaluate_finetuned_voxtell_
# rexgroundingct.py) and then runs it over the official (blind) test set,
# saving one native-resolution prediction NIfTI per volume to
# <run_dir>/official_predictions/ -- for submission to the ReXGroundingCT
# leaderboard. VoxTell counterpart to submit_predict_rexgroundingct_test.sh
# (our own Grounder model's version); same --test-only escape hatch and
# <run_dir>-derived output locations.
#
# Pass --test-only to skip the val-scoring step and only run test
# predictions -- e.g. to retry just the test step after it failed/crashed
# without redoing an already-successful (and much slower) val eval.
#
# Any other extra arguments (e.g. --multi-window) are forwarded to BOTH the
# val-eval and test-predict steps -- must match how --checkpoint was
# fine-tuned.
#
# Usage: sbatch submit_predict_voxtell_rexgroundingct_test.sh [--test-only] <checkpoint> [extra args...]
#   e.g. sbatch submit_predict_voxtell_rexgroundingct_test.sh runs/voxtell_finetune/checkpoints/best.pt
#        sbatch submit_predict_voxtell_rexgroundingct_test.sh --test-only runs/voxtell_finetune/checkpoints/best.pt
#        sbatch submit_predict_voxtell_rexgroundingct_test.sh runs/voxtell_finetune/checkpoints/best.pt --multi-window

TEST_ONLY=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--test-only" ]; then
    TEST_ONLY=1
  else
    ARGS+=("$arg")
  fi
done

CHECKPOINT=${ARGS[0]:?Usage: sbatch submit_predict_voxtell_rexgroundingct_test.sh [--test-only] <checkpoint> [extra args...]}
EXTRA_ARGS=("${ARGS[@]:1}")
RUN_DIR=$(dirname "$(dirname "$(realpath "$CHECKPOINT")")")

DATA_ROOT=/path/to/data
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B  # must be the 4B model -- build_voxtell_model hardcodes text_embedding_dim=2560 to match the pretrained VoxTell checkpoint
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

if [ "$TEST_ONLY" -eq 0 ]; then
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/evaluate_finetuned_voxtell_rexgroundingct.py \
    --checkpoint "$CHECKPOINT" \
    --data-root $DATA_ROOT \
    --text-encoder $TEXT_MODEL_DIR \
    --output "$RUN_DIR/rexgroundingct_val_eval/results.json" \
    "${EXTRA_ARGS[@]}"
fi

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/evaluation/predict_voxtell_rexgroundingct_test.py \
    --checkpoint "$CHECKPOINT" \
    --data-root $DATA_ROOT \
    --text-encoder $TEXT_MODEL_DIR \
    "${EXTRA_ARGS[@]}"
