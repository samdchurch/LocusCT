#!/bin/bash
#SBATCH --job-name=grounder_bench_gz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

DATA_ROOT=/path/to/data/inhouse_abdominal_ct
MODEL_DIR=/path/to/data/models/Qwen3-Embedding-8B
REPO=$HOME/grounder
SIF=$REPO/grounder.sif
HF_CACHE=$HOME/.cache/huggingface

mkdir -p $REPO/logs $HF_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derived from the job ID, not hardcoded -- avoids EADDRINUSE against
# another torch.distributed.run job sharing the node (this job isn't
# --exclusive).
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
    --override model.text_encoder_name=$MODEL_DIR \
      data.train_manifest=official_splits/benchmark_1000_gz.json \
      data.val_manifest=official_splits/benchmark_1000_gz.json \
      data.embedding_cache="" \
      training.num_epochs=2 checkpoint.output_dir=runs/bench_gz
