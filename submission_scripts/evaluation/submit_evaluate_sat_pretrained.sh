#!/bin/bash
#SBATCH --job-name=grounder_sat_eval_pretrained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs the published SAT-Pro checkpoint (github.com/zhaoziheng/SAT, no fine-tuning) on
# the official ONC test set, then the official ED test set, via scripts/evaluation/
# evaluate_sat_onc.py and evaluate_sat_ed.py -- both loaded and run exactly as SAT's own
# inference.py intends (UNET-L vision backbone + BioLORD-based text encoder, DDP-wrapped
# unconditionally by SAT's own build_maskformer()/Text_Encoder(), hence the torchrun
# launch below even for a single GPU). See both scripts' own docstrings for the two
# small, deliberate accommodations needed for this dataset (long free-text referring
# expressions vs. SAT's own short-label filename convention; per-accession "dataset"
# field to avoid output-path collisions) -- neither touches SAT's actual model or
# preprocessing code.
#
# Prerequisite 1: a SEPARATE Singularity image from grounder.sif -- see sat/Dockerfile
# for why (kept separate rather than folded into grounder.sif). Build it the same way
# grounder.sif and biomedparse.sif were built (Dockerfile -> registry -> `singularity
# pull`, since `singularity build --fakeroot`/`--remote` don't work on this cluster):
#   docker build -t <registry>/sat:latest -f sat/Dockerfile sat/
#   docker push <registry>/sat:latest
#   singularity pull sat.sif docker://<registry>/sat:latest
# and place the resulting sat.sif at $REPO/sat.sif (or point SIF below elsewhere).
#
# Prerequisite 2: FremyCompany/BioLORD-2023-C must be pre-downloaded on a machine with
# internet access and staged into the HF cache convention below -- SAT's text encoder
# (sat/SAT/model/knowledge_encoder.py -> model/text_tower.py, and MyTokenizer in
# model/tokenizer.py) calls AutoModel.from_pretrained('FremyCompany/BioLORD-2023-C')
# and AutoTokenizer.from_pretrained(...) on that same repo id at construction time,
# before the checkpoint weights below are loaded on top of it, and this cluster has no
# internet access. E.g. on a machine with internet:
#   python -c "from huggingface_hub import snapshot_download; \
#       snapshot_download('FremyCompany/BioLORD-2023-C')"
# This downloads (by default, with HF_HOME unset) to
# ~/.cache/huggingface/hub/models--FremyCompany--BioLORD-2023-C -- note the "hub/"
# level: huggingface_hub's actual cache root is $HF_HOME/hub, not $HF_HOME itself
# (HF_HUB_CACHE defaults to os.path.join(HF_HOME, "hub")). Upload that
# models--FremyCompany--BioLORD-2023-C directory so it lands at
# $HF_CACHE/hub/models--FremyCompany--BioLORD-2023-C below -- copying it directly into
# $HF_CACHE (skipping the "hub" level) will fail with huggingface_hub's
# LocalEntryNotFoundError despite the snapshot being present.
#
# HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 below force both from_pretrained() calls to
# read straight from that staged cache and skip the Hub-reachability check entirely --
# without them, the library still tries a HEAD request first (matching cache or not),
# which on this cluster's no-internet nodes doesn't fail fast: it hangs through several
# minutes of connection-reset retries before finally falling back to the cache (or
# erroring, if the cache isn't staged yet). Whatever you're pointing this at must
# already be a complete local snapshot -- nothing is fetched at run time.
#
# Prerequisite 3: SAT_Pro.pth and text_encoder.pth staged under the cluster's model
# directory convention (see MODEL_DIR below) -- both available at
# https://huggingface.co/zzh99/SAT.
#
# Usage: sbatch submit_evaluate_sat_pretrained.sh [output_dir]
#   e.g. sbatch submit_evaluate_sat_pretrained.sh outputs/eval/sat_pretrained
#   -> writes outputs/eval/sat_pretrained/onc/results.json
#      and    outputs/eval/sat_pretrained/ed/results.json

OUTPUT_DIR=${1:-outputs/eval/sat_pretrained}

DATA_ROOT=/path/to/data
IMAGE_DIR=$DATA_ROOT/inhouse_abdominal_ct/nifti
ONC_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/labels
ED_MASK_DIR=$DATA_ROOT/inhouse_abdominal_ct/ED_TEST_SET
REPO=$HOME/grounder
SAT_REPO=sat/SAT                                    # relative to /workspace
MODEL_DIR=$DATA_ROOT/models/SAT/Pro
CHECKPOINT=$MODEL_DIR/SAT_Pro.pth
TEXT_ENCODER_CHECKPOINT=$MODEL_DIR/text_encoder.pth
SIF=$REPO/sat.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $REPO/$OUTPUT_DIR/onc $REPO/$OUTPUT_DIR/ed $HF_CACHE

echo "=== ONC official test set ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $HF_CACHE:/cache/huggingface \
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
  $SIF \
  torchrun --nproc_per_node=1 --master_port 29500 \
    scripts/evaluation/evaluate_sat_onc.py \
    --image-dir $IMAGE_DIR \
    --mask-dir $ONC_MASK_DIR \
    --sat-repo $SAT_REPO \
    --checkpoint $CHECKPOINT \
    --text-encoder-checkpoint $TEXT_ENCODER_CHECKPOINT \
    --output "$OUTPUT_DIR/onc/results.json"

echo "=== ED official test set ==="
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $HF_CACHE:/cache/huggingface \
  --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
  $SIF \
  torchrun --nproc_per_node=1 --master_port 29501 \
    scripts/evaluation/evaluate_sat_ed.py \
    --image-dir $IMAGE_DIR \
    --mask-dir $ED_MASK_DIR \
    --sat-repo $SAT_REPO \
    --checkpoint $CHECKPOINT \
    --text-encoder-checkpoint $TEXT_ENCODER_CHECKPOINT \
    --output "$OUTPUT_DIR/ed/results.json"
