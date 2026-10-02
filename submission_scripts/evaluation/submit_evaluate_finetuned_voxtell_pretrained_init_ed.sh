#!/bin/bash
#SBATCH --job-name=grounder_voxtell_pretrained_init_ed_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same evaluate_finetuned_voxtell_ed.py as submit_evaluate_finetuned_voxtell_ed.sh, but
# for a checkpoint fine-tuned from the pretrained VoxTell weights (submit_finetune_
# voxtell.sh, runs/voxtell_finetune by default) instead of --from-scratch: native
# resolution (not the resampled grid) and no --multi-window (finetune_voxtell.py
# defaults --multi-window off unless --from-scratch, or it was passed explicitly at
# training time -- pass it here too via extra args if it was).
#
# Usage: sbatch submit_evaluate_finetuned_voxtell_pretrained_init_ed.sh <checkpoint> [output_dir] [extra args...]
#   e.g. sbatch submit_evaluate_finetuned_voxtell_pretrained_init_ed.sh \
#            runs/voxtell_finetune/checkpoints/best.pt outputs/eval/voxtell_pretrained_init_ed \
#            --tile-overlap 0.5
#
# Any extra arguments beyond checkpoint/output_dir are passed straight
# through to evaluate_finetuned_voxtell_ed.py (e.g. --tile-overlap, --multi-window).
#
# --text-encoder must match whatever was used for the fine-tuning run
# (finetune_voxtell.py's own default is Qwen3-Embedding-4B, not Grounder's 8B).

CHECKPOINT=${1:?Usage: sbatch submit_evaluate_finetuned_voxtell_pretrained_init_ed.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/voxtell_pretrained_init_ed}
shift $(( $# >= 2 ? 2 : $# ))

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/ED_TEST_SET
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
    --image-dir $IMAGE_DIR \
    --mask-dir $MASK_DIR \
    --text-encoder $TEXT_MODEL_DIR \
    --output "$OUTPUT_DIR/results.json" \
    "$@"
