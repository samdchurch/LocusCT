#!/bin/bash
#SBATCH --job-name=grounder_grpimg_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# group_by_image training run: lr 1e-4, no warmup, multi-window
# (lung/soft-tissue/bone) input, gated_cross_attention fusion,
# unet_base_channels=16, 10 epochs, 4x H200 DDP -- on the 192^3
# (2.0x2.0x3.0mm) image/mask grid (nifti_resampled_192/labels_resampled_192,
# from submit_resample_images_192.sh/submit_resample_masks_192.sh), not
# configs/default.yaml's own default 352x352x192 grid.
#
# Switched from 4x L40S (48G/GPU) to 4x H200 (141G/GPU) after an OOM there --
# group_by_image's decoder runs at batch=N (however many findings an image
# has, not a fixed batch_size), so per-step memory varies with the sampled
# image rather than being capped like the old per-triplet path. cpus-per-
# task/mem scaled to submit_h200_continue.sh's own per-GPU ratio (8
# cpus/64G per GPU) for a non-exclusive partial-node H200 request.
#
# training.batch_size=2 is set for the record (embedded in the run
# name/checkpoint dir via _build_run_name) but has NO effect on the actual
# per-step composition under group_by_image=true -- each step is exactly
# one randomly sampled image and ALL of its (phrase, mask) findings, up to
# data.max_findings_per_image, not a fixed images-per-step count. The
# effective number of images processed per optimizer sync across the job is
# world_size (4, one per rank), not 2.
#
# max_findings_per_image=8: per scripts/diagnostics/estimate_max_findings_per_image.py
# on this exact config (4x H200, 139.8 GiB/GPU) -- N=8 probed at 65 GiB
# (47% of one GPU's memory, before optimizer state/DDP overhead the probe
# doesn't include), N=16 at 127 GiB (91%, essentially no margin), N=24 OOM'd.
# Data side: median 1 finding/image, p99=7, so 8 only truncates 0.4% of images.
#
# Usage: sbatch submit_grpimg_h200.sh

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
      data.group_by_image=true \
      data.max_findings_per_image=8 \
      data.multi_window=true \
      training.batch_size=2 \
      training.num_epochs=10 \
      optimizer.lr=1e-4 \
      scheduler.warmup_epochs=0 \
      model.fusion_type=gated_cross_attention \
      model.unet_base_channels=16
