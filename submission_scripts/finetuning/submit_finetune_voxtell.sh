#!/bin/bash
#SBATCH --job-name=grounder_voxtell_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=14:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Fine-tunes the full pretrained VoxTell model (encoder + Qwen3-4B text-fusion
# decoder) end-to-end on our own referring-expression manifests, on the full
# training manifest across 4 H200s (see finetune_voxtell.py's module
# docstring for how this differs from VoxTell's own encoder-only
# transfer-learning path, voxtell-finetune).
#
# Defaults to --embedding-cache $DATA_ROOT/voxtell_embeddings_mmap -- run
# submit_precompute_voxtell_embeddings.sh once first to build it. With the
# cache present, the text backbone is never loaded during fine-tuning at
# all; VoxTellFinetuneDataset fails fast at startup if the cache is missing
# entries for the current manifest(s) (rebuild it if so). Pass
# --embedding-cache "" via extra args to fall back to live encoding instead.
#
# finetune_voxtell.py's DDP path has never been run before -- recommend a
# quick multi-GPU smoke test first to confirm the distributed wiring itself
# works, before trusting it with this job's full 14h budget:
#   sbatch submit_finetune_voxtell.sh --max-samples 20 --num-epochs 2
#
# Caveat: checkpoints only save at the end of a completed epoch (see
# save_checkpoint in finetune_voxtell.py) -- there's no mid-epoch checkpoint
# like train.py's checkpoint_every_n_steps. On the full ~100k-sample manifest
# at batch_size=2 across 4 GPUs, one epoch may not finish inside 14h; if the
# job times out mid-epoch, that epoch's progress is lost and a resubmit would
# redo it from the last completed epoch (via --resume, passed explicitly --
# there's no train.py-style auto-resume-from-latest here either). Ask if you
# want mid-epoch checkpointing added.
#
# finetune_voxtell.py's --random-crop-fraction defaults to 0.33 -- 33% of
# training patches are a truly random 192^3 crop (may be empty or clip the
# mask) instead of foreground-guaranteed, so the model sees the kind of
# tiles real sliding-window eval hands it (see finetune_voxtell.py's module
# docstring's "Random crops" section). Pass --random-crop-fraction 0.0 via
# extra args to restore the old 100%-foreground-guaranteed behavior.
#
# Any extra arguments are passed straight through to finetune_voxtell.py --
# e.g. --batch-size, --num-epochs, --max-samples, --random-crop-fraction, or
# --resume runs/voxtell_finetune/checkpoints/epoch_0010.pt to continue a run.
#
# Usage: sbatch submit_finetune_voxtell.sh [extra finetune_voxtell.py args...]

DATA_ROOT=/path/to/data  # parent of inhouse_abdominal_ct/{nifti,labels} -- native resolution, not the resampled grid configs/default.yaml uses
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B  # must be the 4B model -- build_voxtell_model hardcodes text_embedding_dim=2560 to match the pretrained VoxTell checkpoint
EMBEDDING_CACHE=$DATA_ROOT/voxtell_embeddings_mmap  # from submit_precompute_voxtell_embeddings.sh
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derived from the job ID, not hardcoded -- avoids EADDRINUSE against another
# torch.distributed.run job sharing the node (this job isn't --exclusive, so
# it may share a node with other jobs, unlike submit_h200.sh's 8-GPU job).
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $SIF \
  -m torch.distributed.run --nproc_per_node=4 --master_port=$MASTER_PORT \
  finetune_voxtell.py \
    --text-encoder $TEXT_MODEL_DIR \
    --embedding-cache $EMBEDDING_CACHE \
    "$@"
