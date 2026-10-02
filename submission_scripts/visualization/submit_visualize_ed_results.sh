#!/bin/bash
#SBATCH --job-name=grounder_viz_ed_results
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Usage: sbatch submit_visualize_ed_results.sh <pred_mask_dir> [output_dir]
#   e.g. sbatch submit_visualize_ed_results.sh /path/to/data/temp/ed_test_test/predicted_masks outputs/viz/ed_results

PRED_MASK_DIR=${1:?Usage: sbatch submit_visualize_ed_results.sh <pred_mask_dir> [output_dir]}
OUTPUT_DIR=${2:-outputs/viz/ed_results}

DATA_ROOT=/path/to/data
RAW_IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
GT_MASK_DIR=$DATA_ROOT/ED_EXAMPLES_DATASET/NIFTI_DATA
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind "$PRED_MASK_DIR":"$PRED_MASK_DIR" \
  $SIF \
  scripts/visualization/visualize_ed_results.py \
    --raw-image-dir $RAW_IMAGE_DIR \
    --gt-mask-dir $GT_MASK_DIR \
    --pred-mask-dir "$PRED_MASK_DIR" \
    --output-dir "$OUTPUT_DIR"
