#!/bin/bash
#SBATCH --job-name=grounder_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

echo "Node: $SLURMD_NODENAME"

DATA_ROOT=/path/to/data  # parent of both inhouse_abdominal_ct and ReXGroundingCT
MODEL_DIR=$DATA_ROOT/models/Qwen3-Embedding-8B
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

singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  --bind $MODEL_DIR:$MODEL_DIR \
  --bind $HF_CACHE:/cache/huggingface \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env NCCL_DEBUG=INFO \
  --env NCCL_DEBUG_FILE=$REPO/logs/nccl_%j_rank%r.log \
  --env CC=gcc \
  $SIF \
  -m torch.distributed.run --nproc_per_node=8 --master_port=$MASTER_PORT \
  train.py --config configs/default.yaml \
    --override model.text_encoder_name=$MODEL_DIR \
      checkpoint.output_dir=runs/h200
