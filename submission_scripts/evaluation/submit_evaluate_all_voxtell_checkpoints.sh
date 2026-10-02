#!/bin/bash
#SBATCH --job-name=grounder_voxtell_all_ckpt_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs BOTH evaluate_finetuned_voxtell_ed.py and evaluate_finetuned_voxtell_onc.py
# on every epoch_*.pt checkpoint currently saved under <checkpoints_dir> --
# finetune_voxtell.py prunes to checkpoint.keep_last_n, so this covers
# whatever currently exists on disk, not necessarily every epoch that was
# ever trained. Sequential within one job (reuses this repo's existing
# per-checkpoint scripts as subprocesses rather than duplicating their
# model/eval logic -- simpler, at the cost of reloading the text backbone
# once per checkpoint per test set instead of once total).
#
# Results: outputs/eval/voxtell_finetuned_ed/<checkpoint_stem>/results.json
#          outputs/eval/voxtell_finetuned_onc/<checkpoint_stem>/results.json
#
# Pass --multi-window as an extra arg if the run was fine-tuned with it
# (must match how the checkpoints were actually trained) -- forwarded to
# both eval scripts for every checkpoint.
#
# Usage: sbatch submit_evaluate_all_voxtell_checkpoints.sh [checkpoints_dir] [extra eval args...]
#   e.g. sbatch submit_evaluate_all_voxtell_checkpoints.sh
#        sbatch submit_evaluate_all_voxtell_checkpoints.sh runs/voxtell_finetune/checkpoints
#        sbatch submit_evaluate_all_voxtell_checkpoints.sh runs/voxtell_finetune/checkpoints --multi-window

CKPT_DIR=${1:-runs/voxtell_finetune/checkpoints}   # relative to $REPO
if [ $# -gt 0 ]; then shift; fi
EXTRA_ARGS=("$@")

DATA_ROOT=/path/to/data
ED_IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
ED_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/ED_TEST_SET
ONC_IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
ONC_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

# Names only (not full paths) -- reconstructed as $CKPT_DIR/$NAME below so
# what's passed to --checkpoint stays relative to $REPO (== /workspace
# inside the container via --pwd/--bind), not an unmapped absolute host path.
CHECKPOINT_NAMES=($(ls "$REPO/$CKPT_DIR"/epoch_*.pt 2>/dev/null | xargs -n1 basename | sort))
if [ ${#CHECKPOINT_NAMES[@]} -eq 0 ]; then
  echo "No epoch_*.pt checkpoints found under $CKPT_DIR" >&2
  exit 1
fi
echo "Found ${#CHECKPOINT_NAMES[@]} checkpoint(s) under $CKPT_DIR: ${CHECKPOINT_NAMES[*]}"

for NAME in "${CHECKPOINT_NAMES[@]}"; do
  CKPT_REL="$CKPT_DIR/$NAME"
  STEM="${NAME%.pt}"

  echo "=== $STEM: ED ==="
  singularity run --nv \
    --pwd /workspace \
    --bind $REPO:/workspace \
    --bind $DATA_ROOT:$DATA_ROOT \
    --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
    --bind $HF_CACHE:/cache/huggingface \
    $SIF \
    scripts/evaluation/evaluate_finetuned_voxtell_ed.py \
      --checkpoint "$CKPT_REL" \
      --image-dir $ED_IMAGE_DIR \
      --mask-dir $ED_MASK_DIR \
      --text-encoder $TEXT_MODEL_DIR \
      --output "outputs/eval/voxtell_finetuned_ed/$STEM/results.json" \
      "${EXTRA_ARGS[@]}"

  echo "=== $STEM: ONC ==="
  singularity run --nv \
    --pwd /workspace \
    --bind $REPO:/workspace \
    --bind $DATA_ROOT:$DATA_ROOT \
    --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
    --bind $HF_CACHE:/cache/huggingface \
    $SIF \
    scripts/evaluation/evaluate_finetuned_voxtell_onc.py \
      --checkpoint "$CKPT_REL" \
      --image-dir $ONC_IMAGE_DIR \
      --mask-dir $ONC_MASK_DIR \
      --text-encoder $TEXT_MODEL_DIR \
      --output "outputs/eval/voxtell_finetuned_onc/$STEM/results.json" \
      "${EXTRA_ARGS[@]}"
done

echo "Done. Results under outputs/eval/voxtell_finetuned_{ed,onc}/<checkpoint>/results.json"
