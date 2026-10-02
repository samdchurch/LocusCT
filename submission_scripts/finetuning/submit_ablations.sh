#!/bin/bash
#SBATCH --job-name=grounder_ablations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs every ablation in configs/ablation_studies.yaml one at a time (train on
# 10% of the training data for 10 epochs, then evaluate on val) via
# run_ablations.py. Safe to resubmit after a timeout/preemption -- already
# finished ablations (runs/ablations/<name>/DONE) are skipped, not redone.
#
# Usage: sbatch submit_ablations.sh
#
# text_encoder_finetune_last2 loads the live 8B Qwen backbone (the others use
# the precomputed embedding_cache) -- if it OOMs on a single L40S, rerun just
# that one on an h200 node instead:
#   singularity run --nv ... $SIF run_ablations.py --config configs/default.yaml \
#     --override model.text_encoder_name=$MODEL_DIR
# (it'll skip every ablation already marked DONE and pick up that one)

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT
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
  run_ablations.py --config configs/default.yaml \
    --override model.text_encoder_name=$MODEL_DIR
