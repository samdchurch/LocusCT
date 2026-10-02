#!/bin/bash
#SBATCH --job-name=grounder_results_figure
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_make_results_figure.sh <checkpoint> [extra scripts/visualization/make_results_figure.py args...]
#
#   Rank all candidates by Dice first:
#     sbatch submit_make_results_figure.sh runs/h200/checkpoints/best.pt --rank
#
#   Then build the figure from hand-picked ids:
#     sbatch submit_make_results_figure.sh runs/h200/checkpoints/best.pt \
#       --ed-cases "id1,id2,id3,id4" --onc-cases "id1,id2,id3,id4"

CHECKPOINT=${1:?Usage: sbatch submit_make_results_figure.sh <checkpoint> [extra args...]}
shift

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti_resampled
ED_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/ED_TEST_SET_resampled
ONC_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels_resampled
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/visualization/make_results_figure.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --image-dir $IMAGE_DIR \
    --ed-mask-dir $ED_MASK_DIR \
    --onc-mask-dir $ONC_MASK_DIR \
    --override model.text_encoder_name=$MODEL_DIR \
    "$@"
