#!/bin/bash
#SBATCH --job-name=grounder_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Precomputes Grounder's raw (pre-projection) text-embedding cache for
# configs/default.yaml's own default train_manifest/val_manifest/test_manifest
# (official_splits/all_data_train.json, curated_ed_onc_val_data.json,
# all_test_data.json) -- covers BOTH submit_h200_4gpu.sh (full data) and
# submit_h200_4gpu_30pct.sh (all_data_train_30pct.json is a strict subset of
# all_data_train.json, keyed by mask path, same dedup convention
# precompute_embeddings.py uses) with a single cache; no need to build a
# second one for the 30pct run. curated_ed_onc_val_data.json is also already
# the disjoint union of curated_ed_val_data.json + curated_onc_val_data.json
# (verified separately), so this also covers Trainer.macro_hit_rate_epoch's
# ED/ONC val loaders without needing them listed separately.
#
# Neither submit_h200_4gpu.sh nor submit_h200_4gpu_30pct.sh currently pass
# data.embedding_cache -- they train against the live 8B TextEncoder. Add
# data.embedding_cache=$DATA_ROOT/grounder_embeddings_mmap to their --override
# list once this finishes to actually use it (frees the 8B backbone's GPU
# memory and skips redundant re-encoding every epoch, same benefit as
# submit_precompute_voxtell_embeddings.sh does for VoxTell).
#
# --text-field selects which manifest field to embed: 'sentence' (default, the
# bare finding text) or 'in-context' (the surrounding report section with the
# finding wrapped in <REF></REF> tags) -- see precompute_embeddings.py's module
# docstring. Each field needs its own cache; output_dir defaults to a
# field-specific path so the two don't collide. Note configs/default.yaml's
# data.max_text_len=256 truncates whatever's tokenized here -- fine for plain
# sentences, but long in-context excerpts may get cut off; check
# precompute_embeddings.py's logged output for how many/how badly before
# relying on it.
#
# Non-'sentence' fields aren't populated for every sample (~18% of samples have
# no 'in-context' text) -- for any --text-field other than 'sentence', this
# passes --fallback-field sentence automatically so those samples still get an
# embedding (of their plain sentence) instead of being dropped from the cache.
#
# Usage: sbatch submit_precompute_grounder_embeddings.sh [output_dir] [text_field]
#   e.g. sbatch submit_precompute_grounder_embeddings.sh "" in-context

DATA_ROOT=/path/to/data  # where the cache is written (home dir quota is too small for raw hidden-state caches)
TEXT_FIELD=${2:-sentence}
FALLBACK_ARGS=()
if [ "$TEXT_FIELD" = "sentence" ]; then
  OUTPUT_DIR=${1:-$DATA_ROOT/grounder_embeddings_mmap}
else
  OUTPUT_DIR=${1:-$DATA_ROOT/grounder_${TEXT_FIELD//-/_}_embeddings_mmap}
  FALLBACK_ARGS=(--fallback-field sentence)
fi
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
  scripts/embedding/precompute_embeddings.py --config configs/default.yaml \
    --text-field "$TEXT_FIELD" \
    "${FALLBACK_ARGS[@]}" \
    --output "$OUTPUT_DIR" \
    --override model.text_encoder_name=$MODEL_DIR
