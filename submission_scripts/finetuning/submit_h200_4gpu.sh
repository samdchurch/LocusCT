#!/bin/bash
#SBATCH --job-name=grounder_h200_4gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Plain per-triplet training run on configs/default.yaml's own default grid
# (352x352x192, fixed spatial_mode -- see data.image_dir/mask_dir): lr=1e-4,
# no warmup, batch_size=3, 4x H200 DDP. num_epochs/early_stop_patience left
# at their config defaults (100 / 4) -- this run trains until Macro Hit Rate
# early-stops, not for a fixed epoch count (see training/trainer.py's
# macro_hit_rate_epoch and Trainer.fit).
#
# One epoch per job: train.py is called with --epochs-this-job 1 (val_epoch +
# macro_hit_rate_epoch already run every epoch inside Trainer.fit, so this
# still evaluates each epoch, not just trains it) and auto-resumes from the
# latest checkpoint as usual. training.num_epochs is NOT touched between
# resubmissions -- it's also the cosine LR schedule's horizon (see
# build_scheduler), so changing it per epoch would corrupt the LR curve.
# Trainer.fit() instead reports completion via a TRAINING_COMPLETE marker file
# (see train.py) written once early-stop fires or num_epochs is reached; this
# script checks for that marker and either resubmits itself + scancels itself,
# or stops the chain. A job killed by --time mid-epoch does NOT auto-resubmit
# (the post-command lines never run) -- recover with a manual `sbatch` like
# today, via training.checkpoint_every_n_steps' mid-epoch checkpointing. If
# you deliberately restart this chain after an earlier early-stop (e.g. after
# raising early_stop_patience/num_epochs), delete the stale
# runs/h200_4gpu/<run_name>/checkpoints/TRAINING_COMPLETE first or the first
# job will see it and refuse to resubmit.
#
# Usage: sbatch submit_h200_4gpu.sh

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface
SELF=$REPO/submission_scripts/finetuning/submit_h200_4gpu.sh  # not $0 -- under sbatch that's SLURM's spooled copy, not the checked-out path

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
      checkpoint.output_dir=runs/h200_4gpu \
      training.batch_size=3 \
      optimizer.lr=1e-4 \
      scheduler.warmup_epochs=0

STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo "train.py exited with status $STATUS -- not resubmitting (avoids an infinite resubmit loop on a real crash)."
  exit $STATUS
fi

MARKER=$(find $REPO/runs/h200_4gpu -maxdepth 3 -name TRAINING_COMPLETE 2>/dev/null | head -n 1)
if [ -n "$MARKER" ]; then
  echo "Training complete ($MARKER) -- not resubmitting."
else
  echo "Epoch done, training not yet complete -- resubmitting for the next epoch."
  sbatch "$SELF"
  scancel $SLURM_JOB_ID
fi
