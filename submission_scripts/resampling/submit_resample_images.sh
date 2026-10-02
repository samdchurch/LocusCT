#!/bin/bash
#SBATCH --job-name=nifti_resample_images
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --output=logs/resample_images_%j.out
#SBATCH --error=logs/resample_images_%j.err

DATA_ROOT=/path/to/data/inhouse_abdominal_ct
REPO=$HOME/grounder
SIF=$REPO/grounder.sif

mkdir -p $REPO/logs

echo "Job started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

singularity run \
  --pwd /workspace \
  --bind $REPO:/workspace \
  --bind $DATA_ROOT:$DATA_ROOT \
  $SIF \
  scripts/data_prep/resample_and_crop.py --workers "$SLURM_CPUS_PER_TASK" --force

echo "Job finished: $(date)"
