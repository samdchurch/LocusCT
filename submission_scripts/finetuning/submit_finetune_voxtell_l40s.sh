#!/bin/bash
#SBATCH --job-name=grounder_voxtell_finetune_l40s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Fine-tunes the full pretrained VoxTell model end-to-end on our complete
# training manifest across 4 L40S, with early stopping once Macro Hit Rate
# (mean of ED-category and ONC hit rates against curated_ed_val_data.json/
# curated_onc_val_data.json -- see finetune_voxtell.py's module docstring and
# training/trainer.py's macro_hit_rate_epoch, which this reuses) hasn't
# improved for 4 epochs -- --num-epochs 100 is just an upper-bound safety net
# on the cosine LR schedule, same convention as configs/default.yaml.
#
# --lr 5e-6 is lower than submit_finetune_voxtell.sh's 4x H200 run (default
# 1e-5) -- pass a different --lr via extra args below to override.
#
# L40S has 48GB VRAM vs. the 4x H200 script's 141GB/card, and finetune_
# voxtell.py has no gradient checkpointing -- --batch-size 1 here is an
# UNTESTED conservative guess, not a confirmed-fitting value. Run a quick
# smoke test first to confirm it (or a larger batch size) actually fits and
# to sanity check the DDP wiring, before trusting a job to this time budget:
#   sbatch submit_finetune_voxtell_l40s.sh --max-samples 20 --num-epochs 2
#
# Defaults to --embedding-cache $DATA_ROOT/voxtell_embeddings_mmap -- run
# submit_precompute_voxtell_embeddings.sh once first to build it (its default
# manifests already cover curated_ed_onc_val_data.json, which is the exact
# union of the ED/ONC split manifests used for Macro Hit Rate here, so no
# separate rebuild is needed for those). With the cache present, the text
# backbone is never loaded during fine-tuning at all. Pass
# --embedding-cache "" via extra args to fall back to live encoding instead.
#
# Caveat: checkpoints only save at the end of a completed epoch -- there's no
# mid-epoch checkpoint like train.py's checkpoint_every_n_steps. On the full
# ~100k-sample manifest, one epoch may not finish inside this job's time
# budget; if the job times out mid-epoch, that epoch's progress is lost and a
# resubmit would redo it from the last completed epoch (via --resume, passed
# explicitly -- there's no train.py-style auto-resume-from-latest here).
#
# Pass --multi-window to swap in our own experimental 3-channel (lung/soft-
# tissue/bone) windowed input -- the same scheme configs/default.yaml's own
# model.multi_window=true uses for Grounder's own from-scratch UNet -- instead
# of the single raw-HU Z-score channel the published VoxTell checkpoint was
# trained on (see finetune_voxtell.py's module docstring). Detected below to
# route the run into its own runs/voxtell_finetune_l40s_mw output dir, so a
# --multi-window run's checkpoints never mix with (or get picked up via
# --resume against) a single-channel run's.
#
# finetune_voxtell.py's --random-crop-fraction defaults to 0.33 -- 33% of
# training patches are a truly random 192^3 crop (may be empty or clip the
# mask) instead of foreground-guaranteed, so the model sees the kind of
# tiles real sliding-window eval hands it (see finetune_voxtell.py's module
# docstring's "Random crops" section). Pass --random-crop-fraction 0.0 via
# extra args to restore the old 100%-foreground-guaranteed behavior.
#
# Any extra arguments are passed straight through to finetune_voxtell.py --
# e.g. --batch-size, --lr, --early-stop-patience, --multi-window,
# --random-crop-fraction, or
# --resume runs/voxtell_finetune_l40s/checkpoints/epoch_0010.pt to continue a run.
#
# Usage: sbatch submit_finetune_voxtell_l40s.sh [--multi-window] [extra finetune_voxtell.py args...]

DATA_ROOT=/path/to/data  # parent of inhouse_abdominal_ct/{nifti,labels} -- native resolution, not the resampled grid configs/default.yaml uses
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B  # must be the 4B model -- build_voxtell_model hardcodes text_embedding_dim=2560 to match the pretrained VoxTell checkpoint
EMBEDDING_CACHE=$DATA_ROOT/voxtell_embeddings_mmap  # from submit_precompute_voxtell_embeddings.sh
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

OUTPUT_DIR=runs/voxtell_finetune_l40s
for arg in "$@"; do
  if [ "$arg" = "--multi-window" ]; then
    OUTPUT_DIR=runs/voxtell_finetune_l40s_mw
  fi
done

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derived from the job ID, not hardcoded -- avoids EADDRINUSE against another
# torch.distributed.run job sharing the node (this job isn't --exclusive, so
# it may share a node with other jobs).
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
    --output-dir $OUTPUT_DIR \
    --batch-size 1 \
    --lr 5e-6 \
    --num-epochs 100 \
    --early-stop-patience 4 \
    "$@"
