#!/bin/bash
#SBATCH --job-name=grounder_segvol_eval_pretrained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the published SegVol_v1.pth checkpoint (github.com/BAAI-DCAI/SegVol, no
# fine-tuning) on the official ONC test set, then the official ED test set, via
# scripts/evaluation/evaluate_segvol_onc.py and evaluate_segvol_ed.py -- both loaded
# and run via SegVol's own real preprocessing (process_ct_gt) and its own real
# "zoom-in-zoom-out" inference loop (zoom_in_zoom_out, imported directly from the
# repo's own inference_demo.py), entirely unmodified. See both scripts' own docstrings
# for the two prompt-related decisions made for this evaluation (text-prompt only, not
# the demo's own default text+GT-derived-box combination; CLIP-token-budget truncation
# of the (long, free-text) referring expressions, since SegVol's text encoder is
# templated for short category names and has no truncation set) -- both confirmed with
# the user, and neither touches SegVol's actual model or preprocessing code. Unlike
# SAT, SegVol's demo runs single-process (torch.nn.DataParallel, not DDP), so no
# torchrun launch is needed here.
#
# Prerequisite 1: a SEPARATE Singularity image from grounder.sif/sat.sif/biomedparse.sif
# -- see segvol/Dockerfile for why (SegVol's own pinned monai==0.9.0/transformers==
# 4.18.0 are well behind what grounder.sif ships). Build it the same way the other
# baselines' images were built (Dockerfile -> registry -> `singularity pull`, since
# `singularity build --fakeroot`/`--remote` don't work on this cluster):
#   docker build -t <registry>/segvol:latest -f segvol/Dockerfile segvol/
#   docker push <registry>/segvol:latest
#   singularity pull segvol.sif docker://<registry>/segvol:latest
# and place the resulting segvol.sif at $REPO/segvol.sif (or point SIF below elsewhere).
#
# Prerequisite 2: SegVol_v1.pth staged under the cluster's model directory convention
# (see MODEL_DIR below) -- available at https://huggingface.co/BAAI/SegVol/tree/main
# or the Google Drive link in the repo's README.
#
# Usage: sbatch submit_evaluate_segvol_pretrained.sh [output_dir]
#   e.g. sbatch submit_evaluate_segvol_pretrained.sh outputs/eval/segvol_pretrained
#   -> writes outputs/eval/segvol_pretrained/onc/results.json
#      and    outputs/eval/segvol_pretrained/ed/results.json

OUTPUT_DIR=${1:-outputs/eval/segvol_pretrained}

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
ONC_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels
ED_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/ED_TEST_SET
REPO=$HOME/grounder
SEGVOL_REPO=segvol/SegVol                          # relative to /workspace
CHECKPOINT=$DATA_ROOT/models/SegVol/SegVol_v1.pth
SIF=$REPO/segvol.sif

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR/onc $REPO/$OUTPUT_DIR/ed

echo "=== ONC official test set ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  python scripts/evaluation/evaluate_segvol_onc.py \
    --image-dir $IMAGE_DIR \
    --mask-dir $ONC_MASK_DIR \
    --segvol-repo $SEGVOL_REPO \
    --checkpoint $CHECKPOINT \
    --output "$OUTPUT_DIR/onc/results.json"

echo "=== ED official test set ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  python scripts/evaluation/evaluate_segvol_ed.py \
    --image-dir $IMAGE_DIR \
    --mask-dir $ED_MASK_DIR \
    --segvol-repo $SEGVOL_REPO \
    --checkpoint $CHECKPOINT \
    --output "$OUTPUT_DIR/ed/results.json"
