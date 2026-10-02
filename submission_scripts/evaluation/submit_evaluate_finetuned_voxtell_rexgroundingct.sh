#!/bin/bash
#SBATCH --job-name=grounder_voxtell_rex_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_evaluate_finetuned_voxtell_rexgroundingct.sh <checkpoint> [output_dir]
#   e.g. sbatch submit_evaluate_finetuned_voxtell_rexgroundingct.sh \
#            runs/voxtell_finetune/checkpoints/best.pt outputs/eval/voxtell_finetuned_rex
#
# ReXGroundingCT is an external public benchmark, not part of
# finetune_voxtell.py's own training data -- this measures cross-dataset
# generalization after in-house fine-tuning, not in-domain performance.
#
# --text-encoder must match whatever was used for the fine-tuning run
# (finetune_voxtell.py's own default is Qwen3-Embedding-4B, not Grounder's
# 8B -- see submit_finetune_voxtell.sh). --multi-window must also be passed
# here if the checkpoint was fine-tuned with --multi-window.

CHECKPOINT=${1:?Usage: sbatch submit_evaluate_finetuned_voxtell_rexgroundingct.sh <checkpoint> [output_dir]}
OUTPUT_DIR=${2:-outputs/eval/voxtell_finetuned_rex}

DATA_ROOT=/path/to/data
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
  scripts/evaluation/evaluate_finetuned_voxtell_rexgroundingct.py \
    --checkpoint "$CHECKPOINT" \
    --data-root $DATA_ROOT \
    --text-encoder $TEXT_MODEL_DIR \
    --output "$OUTPUT_DIR/results.json"
