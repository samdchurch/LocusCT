#!/bin/bash
#SBATCH --job-name=grounder_viz_merlin_predictions
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Re-runs scripts/visualization/visualize_merlin_predictions.py over every
# <output_dir>/<category>/predictions.json already on disk -- no inference, just
# visualization. Useful standalone: re-generate viz images after tweaking
# visualize_merlin_predictions.py itself, or to pick up predicted_masks/ files that
# weren't yet visible to submit_predict_merlin.sh's own viz step (this repo's network
# share can have a brief write-visibility lag between separate singularity run
# invocations -- if that step reported "No such file or no access" for files that DO
# exist when you check with `ls` a moment later, just rerun this script; predict_
# merlin.py's inference itself never needs to run again).
#
# Usage: sbatch submit_visualize_merlin_predictions.sh <output_dir>
#   e.g. sbatch submit_visualize_merlin_predictions.sh outputs/eval/merlin_predictions
#   e.g. sbatch submit_visualize_merlin_predictions.sh /path/to/data/merlin_output_findings

OUTPUT_DIR=${1:?Usage: sbatch submit_visualize_merlin_predictions.sh <output_dir>}
# Singularity's --bind requires an absolute destination path -- resolve up front so a
# relative $OUTPUT_DIR (e.g. "runs/...") doesn't hit "destination must be an absolute
# path" below. Also collapses any trailing/double slashes.
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

DATA_ROOT=/path/to/data  # predictions.json's image_path values live under here (merlinabdominalctdataset)
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

for category_dir in "$OUTPUT_DIR"/*/; do
  predictions_file="$category_dir/predictions.json"
  if [ -f "$predictions_file" ]; then
    echo "Visualizing $predictions_file"
    singularity run \
      --pwd /workspace \
      --bind $REPO:/workspace \
      --bind $DATA_ROOT:$DATA_ROOT \
      --bind "$OUTPUT_DIR":"$OUTPUT_DIR" \
      $SIF \
      scripts/visualization/visualize_merlin_predictions.py \
        --predictions "$predictions_file" \
        --n 0 \
        --output-dir "$category_dir/viz"
  fi
done
