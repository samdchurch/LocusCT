#!/bin/bash
#SBATCH --job-name=grounder_all_baselines_eval_pretrained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs all four ORIGINAL, published (no fine-tuning) baseline checkpoints --
# BiomedParse-v2, SAT-Pro, SegVol, VoxTell -- on the official ONC test set and
# the official ED test set, sequentially within one job. Reuses this repo's
# existing per-model submission scripts as subprocesses rather than
# duplicating their singularity/torchrun invocations (each one's own
# #SBATCH header is inert when the script is run directly via `bash` instead
# of `sbatch`, so this is safe):
#   submit_evaluate_biomedparse_pretrained.sh
#   submit_evaluate_sat_pretrained.sh
#   submit_evaluate_segvol_pretrained.sh
#   submit_evaluate_voxtell_pretrained.sh
#
# Each of those four has its own prerequisites (a separate Singularity image
# plus staged checkpoint(s) -- see that script's own header comment) that
# must be set up beforehand. If a model isn't set up yet, or its run errors
# out for any other reason (OOM, bad path, etc.), this script logs it and
# moves on to the next model rather than aborting the whole job -- that
# model just shows as MISSING/ERROR in the final report instead of blocking
# the others.
#
# Time limit is sized for all four run back-to-back (BiomedParse alone
# budgets 16h per its own script's time limit; the other three 8h each --
# see submit_evaluate_biomedparse_pretrained.sh's header for why it's
# slower). If you have four GPUs available, it's faster to `sbatch` the four
# individual scripts separately instead of this one, then run
# report_baseline_results.py once they've all finished.
#
# Usage: sbatch submit_evaluate_all_baselines_pretrained.sh [output_dir]
#   e.g. sbatch submit_evaluate_all_baselines_pretrained.sh outputs/eval/all_baselines_pretrained
#   -> writes outputs/eval/all_baselines_pretrained/<model>/{onc,ed}/results.json
#      and    outputs/eval/all_baselines_pretrained/summary.txt

OUTPUT_DIR=${1:-outputs/eval/all_baselines_pretrained}

REPO=$HOME/grounder
SIF=$REPO/grounder.sif   # only used for the final report step below (stdlib-only, just keeps the python version consistent with the rest of the repo)
SCRIPT_DIR=$REPO/submission_scripts/evaluation

mkdir -p $REPO/logs

declare -A STATUS

run_baseline() {
  local name=$1 script=$2
  echo ""
  echo "########## $name ##########"
  if bash "$script" "$OUTPUT_DIR/$name"; then
    STATUS[$name]="completed (see report below for per-split pass/fail)"
  else
    STATUS[$name]="FAILED (exit $?) -- skipped, continuing with remaining baselines"
    echo "!!! $name FAILED -- continuing with remaining baselines !!!" >&2
  fi
}

run_baseline biomedparse "$SCRIPT_DIR/submit_evaluate_biomedparse_pretrained.sh"
run_baseline sat         "$SCRIPT_DIR/submit_evaluate_sat_pretrained.sh"
run_baseline segvol      "$SCRIPT_DIR/submit_evaluate_segvol_pretrained.sh"
run_baseline voxtell     "$SCRIPT_DIR/submit_evaluate_voxtell_pretrained.sh"

echo ""
echo "########## Job status per baseline ##########"
for name in biomedparse sat segvol voxtell; do
  echo "$name: ${STATUS[$name]}"
done

echo ""
echo "########## Results report ##########"
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  $SIF \
  python scripts/evaluation/report_baseline_results.py "$OUTPUT_DIR"
