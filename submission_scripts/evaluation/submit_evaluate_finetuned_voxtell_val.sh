#!/bin/bash
#SBATCH --job-name=grounder_voxtell_val_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_evaluate_finetuned_voxtell_val.sh <checkpoint> [output_dir] [extra evaluate_finetuned_voxtell_val.py args...]
#   e.g. sbatch submit_evaluate_finetuned_voxtell_val.sh \
#            runs/voxtell_finetune/checkpoints/best.pt outputs/eval/voxtell_finetuned_val
#   e.g. with overlapping sliding-window tiles:
#        sbatch submit_evaluate_finetuned_voxtell_val.sh \
#            runs/voxtell_finetune/checkpoints/best.pt outputs/eval/voxtell_finetuned_val_overlap \
#            --tile-overlap 0.5
#
# --text-encoder must match whatever was used for the fine-tuning run
# (finetune_voxtell.py's own default is Qwen3-Embedding-4B, not Grounder's
# 8B -- see submit_finetune_voxtell.sh). --multi-window must also be passed
# here if the checkpoint was fine-tuned with --multi-window.
#
# IMAGE_DIR/MASK_DIR below point at the resampled (nifti_resampled/
# labels_resampled) grid to match the --from-scratch VoxTell checkpoint's own
# training input (see finetune_voxtell.py's "Image resolution" docstring
# section) -- switch back to the native inhouse_abdominal_ct/nifti|labels dirs to
# evaluate a checkpoint fine-tuned from the pretrained VoxTell weights
# instead, which trains on native resolution.
#
# Capped to the first 200 cases (--max-samples) for a quick sanity check,
# not a full val pass -- raise/drop it for the complete curated val set.
# Any arguments beyond <checkpoint>/[output_dir] are passed straight through
# to evaluate_finetuned_voxtell_val.py (e.g. --tile-overlap, --max-samples to
# override the 200-case cap below, --visualize).

CHECKPOINT=${1:?Usage: sbatch submit_evaluate_finetuned_voxtell_val.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/voxtell_finetuned_val}
shift $(( $# < 2 ? $# : 2 ))

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti_resampled
MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels_resampled
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
  scripts/evaluation/evaluate_finetuned_voxtell_val.py \
    --checkpoint "$CHECKPOINT" \
    --image-dir $IMAGE_DIR \
    --mask-dir $MASK_DIR \
    --text-encoder $TEXT_MODEL_DIR \
    --multi-window \
    --max-samples 200 \
    --output "$OUTPUT_DIR/results.json" \
    "$@"
