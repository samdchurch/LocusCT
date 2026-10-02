#!/bin/bash
#SBATCH --job-name=grounder_sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Diagnoses a job that dies ~1 minute in with NOTHING written to .out/.err --
# that symptom usually means either (a) the crash happens before anything
# ever gets a chance to print (a bad #SBATCH directive, an unschedulable
# --gres/--exclusive request, a bind-mount path that doesn't exist), (b)
# stdout is buffered and lost on an abrupt kill (OOM, walltime, node fault --
# `set -x` below forces every command to echo immediately, unbuffered), or
# (c) this repo's already-documented network-share write-visibility lag
# between separate singularity invocations (see submit_predict_merlin.sh) --
# .out/.err get written but reading them back too soon shows nothing.
#
# This script logs a timestamped line after every stage to TWO places: SLURM's
# own .out (via stdout) AND a separate marker file written directly under
# logs/, bypassing SLURM's redirect entirely. If the marker file also comes
# back empty/missing, the problem is upstream of anything this repo
# controls -- check `sacct -j <jobid> --format=JobID,State,ExitCode,NodeList,Elapsed,Reason`
# and `scontrol show job <jobid>` (while still pending/running) for SLURM's
# own account. If the marker file has content but .out/.err don't, that
# points at (c) -- wait a bit and re-check, or read logs/sanity_<jobid>.marker
# instead.
#
# Each stage is independent and stops the script on first failure (set -e),
# so whichever STAGE line is the last one logged tells you where it broke.
#
# Adjust --gres/--partition/--exclusive above to match whichever job type is
# actually failing if this generic single-GPU version doesn't reproduce it.
#
# Usage: sbatch submit_sanity_check.sh

set -ex   # -x: echo every command as it runs; -e: stop at first failing command

REPO=$HOME/grounder
SIF=$REPO/grounder.sif
MARKER=$REPO/logs/sanity_${SLURM_JOB_ID}.marker

mkdir -p "$REPO/logs"
: > "$MARKER"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1" | tee -a "$MARKER"
}

log "STAGE 0: job started on host=$(hostname) user=$(whoami) pwd=$(pwd) SLURM_JOB_ID=$SLURM_JOB_ID"

log "STAGE 1: repo/image visible on this node's filesystem"
ls -la "$REPO" >/dev/null
ls -la "$SIF"
log "STAGE 1: OK"

log "STAGE 2: GPU visible to the host (outside the container)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
log "STAGE 2: OK"

log "STAGE 3: singularity container launches at all (no GPU)"
singularity exec "$SIF" echo "container alive"
log "STAGE 3: OK"

log "STAGE 4: singularity --nv GPU passthrough into the container"
singularity exec --nv "$SIF" nvidia-smi --query-gpu=name --format=csv
log "STAGE 4: OK"

log "STAGE 5: python + torch + CUDA inside the container"
singularity exec --nv "$SIF" python -c "
import torch
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device='cuda')
    print('matmul ok:', (x @ x).sum().item())
"
log "STAGE 5: OK"

log "STAGE 6: all sanity checks passed"
