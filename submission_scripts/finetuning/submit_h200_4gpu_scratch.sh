#!/bin/bash
#SBATCH --job-name=grounder_h200_4gpu_scratch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same self-resubmitting chain as submit_h200_4gpu.sh (identical recipe:
# batch_size=3, lr=1e-4, no warmup, configs/default.yaml's own defaults for
# everything else -- spatial_mode=fixed/spatial_size=[352,352,192],
# multi_window=true, unet_base_channels=16, fusion_type=gated_cross_attention
# -- see train.py's _build_run_name, which derives the run's directory name,
# "ch16_bs3_lr1e-04_fixed352x352x192_gated_cross_attention_mw", from exactly
# these values), EXCEPT checkpoint.output_dir points at the absolute scratch
# path this run's checkpoints actually live under
# ($DATA_ROOT/runs/h200_4gpu) instead of submit_h200_4gpu.sh's repo-relative
# runs/h200_4gpu. That script has no "$@" passthrough to override this from
# the command line, and its own TRAINING_COMPLETE marker check looks under
# $REPO/runs/h200_4gpu -- both needed a real copy, not a one-off override.
#
# train.py auto-resumes from the latest checkpoint under
# <checkpoint.output_dir>/<run_name>/checkpoints/ whenever --resume isn't
# passed (mid_epoch.pt if present, else the latest epoch_*.pt) -- see
# _find_latest_checkpoint. As long as the overrides below keep producing the
# same run_name, this picks up exactly where the existing scratch run left
# off; DATA_ROOT (which covers the scratch output_dir) is already bound
# below, so no new bind is needed.
#
# One epoch per job, same as submit_h200_4gpu.sh: train.py is called with
# --epochs-this-job 1 (val_epoch/macro_hit_rate_epoch still run every epoch
# inside Trainer.fit) and this script resubmits itself + scancels the current
# job after each epoch, until a TRAINING_COMPLETE marker appears under this
# run's checkpoint dir on scratch (written once early-stop fires or
# training.num_epochs is reached -- see train.py). A job killed by --time
# mid-epoch does NOT auto-resubmit (the post-command lines never run) --
# recover with a manual `sbatch` like today, via
# training.checkpoint_every_n_steps' mid-epoch checkpointing. training.
# num_epochs is NOT touched between resubmissions -- it's also the cosine LR
# schedule's horizon (see build_scheduler), so changing it per epoch would
# corrupt the LR curve.
#
# Usage: sbatch submit_h200_4gpu_scratch.sh

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
OUTPUT_DIR=$DATA_ROOT/runs/h200_4gpu
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface
SELF=$REPO/submission_scripts/finetuning/submit_h200_4gpu_scratch.sh  # not $0 -- under sbatch that's SLURM's spooled copy, not the checked-out path

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derived from the job ID, not hardcoded -- avoids EADDRINUSE against a
# leftover process from a previous job on this node (this job is
# --exclusive, so it won't collide with another concurrent job's port).
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
    --epochs-this-job 1 \
    --override model.text_encoder_name=$MODEL_DIR \
      checkpoint.output_dir=$OUTPUT_DIR \
      training.batch_size=3 \
      optimizer.lr=1e-4 \
      scheduler.warmup_epochs=0

STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo "train.py exited with status $STATUS -- not resubmitting (avoids an infinite resubmit loop on a real crash)."
  exit $STATUS
fi

MARKER=$(find $OUTPUT_DIR -maxdepth 3 -name TRAINING_COMPLETE 2>/dev/null | head -n 1)
if [ -n "$MARKER" ]; then
  echo "Training complete ($MARKER) -- not resubmitting."
else
  echo "Epoch done, training not yet complete -- resubmitting for the next epoch."
  sbatch "$SELF"
  scancel $SLURM_JOB_ID
fi
