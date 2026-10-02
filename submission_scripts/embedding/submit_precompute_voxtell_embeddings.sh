#!/bin/bash
#SBATCH --job-name=grounder_voxtell_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Precomputes VoxTell's pooled text-embedding cache for finetune_voxtell.py's
# train+val manifests (~100k samples on the full training manifest -- one
# forward pass through the 4B backbone per unique sentence, not batched, so
# this can run long; --time is a generous first guess, adjust once you've
# seen actual throughput).
#
# Run once, then point submit_finetune_voxtell.sh (already wired up) at the
# resulting directory via --embedding-cache -- the text backbone is then
# never loaded during fine-tuning at all.
#
# --text-field selects which manifest field to embed: 'sentence' (default, the
# bare finding text) or 'in-context' (the surrounding report section with the
# finding wrapped in <REF></REF> tags) -- see precompute_voxtell_embeddings.py's
# module docstring. Each field needs its own cache; output_dir defaults to a
# field-specific path so the two don't collide.
#
# Non-'sentence' fields aren't populated for every sample (~18% of samples have
# no 'in-context' text) -- for any --text-field other than 'sentence', this
# passes --fallback-field sentence automatically so those samples still get an
# embedding (of their plain sentence) instead of being dropped from the cache,
# which would otherwise make VoxTellFinetuneDataset refuse to start (it requires
# every manifest sample to have a cache entry).
#
# Usage: sbatch submit_precompute_voxtell_embeddings.sh [output_dir] [text_field]
#   e.g. sbatch submit_precompute_voxtell_embeddings.sh "" in-context

DATA_ROOT=/path/to/data  # where the cache is written (home dir quota is too small for raw hidden-state caches)
TEXT_FIELD=${2:-sentence}
FALLBACK_ARGS=()
if [ "$TEXT_FIELD" = "sentence" ]; then
  OUTPUT_DIR=${1:-$DATA_ROOT/voxtell_embeddings_mmap}
else
  OUTPUT_DIR=${1:-$DATA_ROOT/voxtell_${TEXT_FIELD//-/_}_embeddings_mmap}
  FALLBACK_ARGS=(--fallback-field sentence)
fi
TEXT_MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-4B  # must be the 4B model -- see finetune_voxtell.py
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $TEXT_MODEL_DIR:$TEXT_MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  $SIF \
  scripts/embedding/precompute_voxtell_embeddings.py \
    --text-encoder $TEXT_MODEL_DIR \
    --text-field "$TEXT_FIELD" \
    "${FALLBACK_ARGS[@]}" \
    --output "$OUTPUT_DIR"
