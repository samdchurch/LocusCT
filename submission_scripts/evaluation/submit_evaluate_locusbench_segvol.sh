#!/bin/bash
#SBATCH --job-name=grounder_locusbench_segvol
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the published SegVol_v1.pth checkpoint (github.com/BAAI-DCAI/SegVol,
# no fine-tuning) on LocusBench-Onc, then LocusBench-ED, which replace the
# old official ONC/ED test sets as the official protocol
# (submit_evaluate_segvol_pretrained.sh still works against the old sets,
# just isn't "official" anymore). See that script's own comments for
# prerequisites (separate segvol.sif, SegVol_v1.pth) -- all identical here,
# none of that changes for LocusBench.
#
# IMAGE_DIR/MASK_DIR both point at each LocusBench split's own raw root
# (unresampled) -- SegVol does its own real preprocessing (process_ct_gt) and
# zoom-in-zoom-out inference from raw NIfTI, same as it already does against
# the old official test sets.

OUTPUT_DIR=${1:-outputs/eval/locusbench_segvol}

DATA_ROOT=/path/to/data
ONC_MANIFEST=$DATA_ROOT/LocusBench/LocusBench-Onc/LocusBench-Onc.json
ED_MANIFEST=$DATA_ROOT/LocusBench/LocusBench-ED/LocusBench-ED.json
ONC_DIR=$DATA_ROOT/LocusBench/LocusBench-Onc
ED_DIR=$DATA_ROOT/LocusBench/LocusBench-ED
REPO=$HOME/grounder
SEGVOL_REPO=segvol/SegVol                          # relative to /workspace
CHECKPOINT=$DATA_ROOT/models/SegVol/SegVol_v1.pth
SIF=$REPO/segvol.sif

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR/onc $REPO/$OUTPUT_DIR/ed

echo "=== LocusBench-Onc ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  python scripts/evaluation/evaluate_segvol_onc.py \
    --manifest $ONC_MANIFEST \
    --image-dir $ONC_DIR \
    --mask-dir $ONC_DIR \
    --segvol-repo $SEGVOL_REPO \
    --checkpoint $CHECKPOINT \
    --output "$OUTPUT_DIR/onc/results.json"

echo "=== LocusBench-ED ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  python scripts/evaluation/evaluate_segvol_ed.py \
    --manifest $ED_MANIFEST \
    --image-dir $ED_DIR \
    --mask-dir $ED_DIR \
    --segvol-repo $SEGVOL_REPO \
    --checkpoint $CHECKPOINT \
    --output "$OUTPUT_DIR/ed/results.json"
