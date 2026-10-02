#!/bin/bash
#SBATCH --job-name=grounder_merlin_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT; also where the cache is written (home dir quota is too small for raw embedding caches)
MODEL_DIR=$DATA_ROOT/models/Merlin  # staged ahead of time -- see project plan's one-time staging step. Not the Merlin Abdominal CT Dataset (that's $DATA_ROOT/public_datasets/merlinabdominalctdataset)
CLINICAL_LONGFORMER_DIR=$DATA_ROOT/models/Clinical-Longformer  # save_pretrained() copy of yikuan8/Clinical-Longformer, staged ahead of time -- this cluster has no internet access at all, and merlin's TextEncoder hardcodes that HF Hub repo id with no override
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface
TORCH_CACHE=$HOME/.cache/torch  # avoids re-downloading torchvision's resnet152 ImageNet init every job (its weights are immediately overwritten by Merlin's own checkpoint, but the download still happens unless cached)

mkdir -p $REPO/logs $HF_CACHE $TORCH_CACHE

# The container image is a read-only squashfs, so merlin.Merlin() can't download/write
# its checkpoint into the installed package's own checkpoints/ dir at job time -- instead
# we bind $MODEL_DIR directly onto that in-container path. Resolved dynamically (rather
# than hardcoding a conda/python-version-specific path) so a future grounder.sif rebuild
# with a different base image doesn't silently break this.
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
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $MODEL_DIR:$CHECKPOINTS_DIR \
  --bind $CLINICAL_LONGFORMER_DIR:$CLINICAL_LONGFORMER_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --bind $TORCH_CACHE:/cache/torch \
  $SIF \
  scripts/embedding/precompute_merlin_image_embeddings.py --config configs/default.yaml \
    --merlin-model-dir $MODEL_DIR \
    --clinical-longformer-dir $CLINICAL_LONGFORMER_DIR \
    --output $DATA_ROOT/merlin_image_embeddings_mmap \
    --override \
      data.image_dir=$DATA_ROOT/inhouse_abdominal_ct/nifti \
      data.train_manifest='[official_splits/all_data_train.json,official_splits/all_train_val_data.json,official_splits/curated_ed_train_data.json,official_splits/curated_ed_train_val_data.json,official_splits/curated_onc_train_data.json,official_splits/curated_onc_train_val_data.json]' \
      data.val_manifest='[official_splits/curated_ed_val_data.json,official_splits/curated_onc_val_data.json]' \
      data.test_manifest='[official_splits/all_test_data.json]'
      # ReXGroundingCT_{train,val}.json excluded for now: its "image" paths point into
      # ReXGroundingCT/resampled/... (already resampled once for our pipeline, unlike
      # inhouse_abdominal_ct's nifti/ native originals), and its true pre-resample source
      # (resample_rexgroundingct.py's IMAGES_ROOT) uses a /path/to/data/... path that
      # doesn't match this cluster's /mnt/scratch/... layout -- confirm the actual
      # native ReXGroundingCT path on our cluster before adding it back here.
      #
      # data.image_dir above is overridden to inhouse_abdominal_ct's *native* (pre-resample)
      # nifti/ directory, not the default.yaml nifti_resampled/ used for training --
      # Merlin does its own resampling (RAS, 1.5x1.5x3mm) and shouldn't be fed images
      # already resampled for our own pipeline's spatial mode.
