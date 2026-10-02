#!/bin/bash
#SBATCH --job-name=grounder_locusbench_sat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the published SAT-Pro checkpoint (github.com/zhaoziheng/SAT, no
# fine-tuning) on LocusBench-Onc, then LocusBench-ED, which replace the old
# official ONC/ED test sets as the official protocol
# (submit_evaluate_sat_pretrained.sh still works against the old sets, just
# isn't "official" anymore). See that script's own comments for the DDP
# launch, prerequisites (separate sat.sif, pre-staged BioLORD-2023-C HF
# cache, SAT_Pro.pth/text_encoder.pth) -- all identical here, none of that
# changes for LocusBench.
#
# IMAGE_DIR/MASK_DIR both point at each LocusBench split's own raw root
# (unresampled) -- SAT does its own monai preprocessing (Spacingd(1,1,3) +
# CropForegroundd) from raw NIfTI, same as it already does against the old
# official test sets.

OUTPUT_DIR=${1:-outputs/eval/locusbench_sat}

DATA_ROOT=/path/to/data
ONC_MANIFEST=$DATA_ROOT/LocusBench/LocusBench-Onc/LocusBench-Onc.json
ED_MANIFEST=$DATA_ROOT/LocusBench/LocusBench-ED/LocusBench-ED.json
ONC_DIR=$DATA_ROOT/LocusBench/LocusBench-Onc
ED_DIR=$DATA_ROOT/LocusBench/LocusBench-ED
REPO=$HOME/grounder
SAT_REPO=sat/SAT                                    # relative to /workspace
MODEL_DIR=$DATA_ROOT/models/SAT/Pro
CHECKPOINT=$MODEL_DIR/SAT_Pro.pth
TEXT_ENCODER_CHECKPOINT=$MODEL_DIR/text_encoder.pth
SIF=$REPO/sat.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR/onc $REPO/$OUTPUT_DIR/ed $HF_CACHE

echo "=== LocusBench-Onc ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $HF_CACHE:/cache/huggingface \
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
  $SIF \
  torchrun --nproc_per_node=1 --master_port 29500 \
    scripts/evaluation/evaluate_sat_onc.py \
    --manifest $ONC_MANIFEST \
    --image-dir $ONC_DIR \
    --mask-dir $ONC_DIR \
    --sat-repo $SAT_REPO \
    --checkpoint $CHECKPOINT \
    --text-encoder-checkpoint $TEXT_ENCODER_CHECKPOINT \
    --output "$OUTPUT_DIR/onc/results.json"

echo "=== LocusBench-ED ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $HF_CACHE:/cache/huggingface \
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
  $SIF \
  torchrun --nproc_per_node=1 --master_port 29501 \
    scripts/evaluation/evaluate_sat_ed.py \
    --manifest $ED_MANIFEST \
    --image-dir $ED_DIR \
    --mask-dir $ED_DIR \
    --sat-repo $SAT_REPO \
    --checkpoint $CHECKPOINT \
    --text-encoder-checkpoint $TEXT_ENCODER_CHECKPOINT \
    --output "$OUTPUT_DIR/ed/results.json"
