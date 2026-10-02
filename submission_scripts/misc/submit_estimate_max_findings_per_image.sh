#!/bin/bash
#SBATCH --job-name=grounder_estimate_maxfindings
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Estimates a safe data.max_findings_per_image: prints the real findings-
# per-image histogram from the manifest, then probes real GPU peak memory
# (forward+backward, no actual image/mask files touched -- synthetic
# tensors) at increasing finding counts N on THIS job's GPU. See
# scripts/diagnostics/estimate_max_findings_per_image.py's own docstring.
#
# Pass extra config overrides matching your real training run (fusion_type,
# unet_base_channels, spatial_size, multi_window, etc.) as BARE key=value
# pairs -- no "--override" prefix, they're appended to this script's own
# already-open --override list (repeating --override would just replace
# model.text_encoder_name=$MODEL_DIR below rather than add to it, since
# argparse's nargs="*" doesn't accumulate across repeated flags). Other
# estimate_max_findings_per_image.py flags (--skip-memory-probe, --manifest,
# --ns, ...) aren't supported through this wrapper -- run the script
# directly inside an interactive singularity shell for those.
#
# Defaults to H200 since that's what submit_grpimg_h200.sh currently runs
# on -- match your actual job's GPU type/partition here for the memory
# numbers to be meaningful.
#
# Usage: sbatch submit_estimate_max_findings_per_image.sh [key=value ...]
#   e.g. sbatch submit_estimate_max_findings_per_image.sh \
#          data.group_by_image=true data.multi_window=true \
#          data.spatial_size=[192,192,192] model.fusion_type=gated_cross_attention \
#          model.unet_base_channels=16

DATA_ROOT=/path/to/data
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
  scripts/diagnostics/estimate_max_findings_per_image.py \
    --config configs/default.yaml \
    --override model.text_encoder_name=$MODEL_DIR "$@"
