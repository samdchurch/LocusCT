#!/bin/bash
#SBATCH --job-name=grounder_merlin_encoder_4gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

echo "Node: $SLURMD_NODENAME"

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
MERLIN_MODEL_DIR=$DATA_ROOT/models/Merlin  # staged ahead of time -- see scripts/embedding/precompute_merlin_image_embeddings.py's own staging step. Not the Merlin Abdominal CT Dataset (that's $DATA_ROOT/public_datasets/merlinabdominalctdataset)
CLINICAL_LONGFORMER_DIR=$DATA_ROOT/models/Clinical-Longformer  # save_pretrained() copy of yikuan8/Clinical-Longformer, staged ahead of time -- this cluster has no internet access, and merlin's TextEncoder hardcodes that HF Hub repo id with no override
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$REPO/logs/nccl_%j_rank%r.log

# Derived from the job ID, not hardcoded -- avoids EADDRINUSE against a
# leftover process from a previous job on this node (this job is
# --exclusive, so it won't collide with another concurrent job's port).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

# merlin.Merlin() can't download/write its checkpoint into the installed
# package's own checkpoints/ dir at job time (read-only container image) --
# bind $MERLIN_MODEL_DIR directly onto that in-container path instead.
# Resolved dynamically (not hardcoded) so a future grounder.sif rebuild with
# a different base image doesn't silently break this -- same approach as
# submit_precompute_merlin_image_embeddings.sh.
CHECKPOINTS_DIR=$(singularity exec $SIF python -c \
  "import os, merlin.models.load as m; print(os.path.join(os.path.dirname(os.path.abspath(m.__file__)), 'checkpoints'))")
if [ -z "$CHECKPOINTS_DIR" ]; then
  echo "Failed to resolve merlin's in-container checkpoints directory" >&2
  exit 1
fi

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $MERLIN_MODEL_DIR:$MERLIN_MODEL_DIR \
  --bind $MERLIN_MODEL_DIR:$CHECKPOINTS_DIR \
  --bind $CLINICAL_LONGFORMER_DIR:$CLINICAL_LONGFORMER_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env NCCL_DEBUG=INFO \
  --env NCCL_DEBUG_FILE=$REPO/logs/nccl_%j_rank%r.log \
  --env CC=gcc \
  $SIF \
  -m torch.distributed.run --nproc_per_node=4 --master_port=$MASTER_PORT \
  train.py --config configs/default.yaml \
    --override model.text_encoder_name=$TEXT_MODEL_DIR \
      model.encoder_type=merlin \
      model.merlin.model_dir=$MERLIN_MODEL_DIR \
      model.merlin.clinical_longformer_dir=$CLINICAL_LONGFORMER_DIR \
      data.merlin.image_dir=$DATA_ROOT/inhouse_abdominal_ct/nifti \
      data.train_manifest='[official_splits/all_data_train.json,official_splits/all_train_val_data.json,official_splits/curated_ed_train_data.json,official_splits/curated_ed_train_val_data.json,official_splits/curated_onc_train_data.json,official_splits/curated_onc_train_val_data.json]' \
      data.val_manifest='[official_splits/curated_ed_val_data.json,official_splits/curated_onc_val_data.json]' \
      checkpoint.output_dir=runs/merlin_encoder_4gpu
      # ReXGroundingCT excluded from both manifests -- its native (pre-resample)
      # source path isn't confirmed on this cluster yet (same caveat as
      # submit_precompute_merlin_image_embeddings.sh). data.merlin.image_dir
      # is the *native* inhouse_abdominal_ct/nifti dir, not nifti_resampled/ --
      # Merlin does its own RAS/1.5x1.5x3mm resampling and shouldn't be fed
      # images already resampled for the plain-UNet pipeline's spatial_mode.
