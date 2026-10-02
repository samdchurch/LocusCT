#!/bin/bash
#SBATCH --job-name=grounder_mw_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Plain per-triplet training run (batch_size=6 random (image,phrase,mask)
# triplets/step, NOT group_by_image -- see submit_grpimg_h200.sh, paused
# for now): lr 1e-4, no warmup, multi-window (lung/soft-tissue/bone) input,
# gated_cross_attention fusion, unet_base_channels=16, 10 epochs, 4x H200
# DDP -- on the 192^3 (2.0x2.0x3.0mm) image/mask grid
# (nifti_resampled_192/labels_resampled_192, from
# submit_resample_images_192.sh/submit_resample_masks_192.sh), not
# configs/default.yaml's own default 352x352x192 grid.
#
# 4x H200, not L40S: group_by_image's OOM on L40S was with variable
# (uncapped at the time) per-step decoder batch size, not this plain fixed
# batch_size=6 path -- untested at this batch_size/grid/multi_window combo,
# so keeping the GPU with headroom rather than assuming L40S is fine here
# too. If you want to try L40S, scripts/diagnostics/
# estimate_max_findings_per_image.py's memory-probe half isn't directly
# reusable (it probes group_by_image's shared-encoder N-findings path, not
# a plain fixed-batch_size one) -- would need a small adaptation.
#
# Usage: sbatch submit_mw_h200.sh

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti_resampled_192
MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels_resampled_192
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derived from the job ID, not hardcoded -- avoids EADDRINUSE against
# another torch.distributed.run job sharing the node (this job isn't
# --exclusive).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $SIF \
  -m torch.distributed.run --nproc_per_node=4 --master_port=$MASTER_PORT \
  train.py --config configs/default.yaml \
    --override model.text_encoder_name=$MODEL_DIR \
      data.image_dir=$IMAGE_DIR \
      data.mask_dir=$MASK_DIR \
      data.spatial_size=[192,192,192] \
      data.group_by_image=false \
      data.multi_window=true \
      training.batch_size=6 \
      training.num_epochs=10 \
      optimizer.lr=1e-4 \
      scheduler.warmup_epochs=0 \
      model.fusion_type=gated_cross_attention \
      model.unet_base_channels=16
