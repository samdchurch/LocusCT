#!/bin/bash
#SBATCH --job-name=grounder_all_locusbench_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs all six models -- Grounder, finetuned VoxTell, and the four published
# (no fine-tuning) baselines (BiomedParse-v2, SAT-Pro, SegVol, pretrained
# VoxTell) -- on LocusBench-Onc and LocusBench-ED sequentially within one
# job, then aggregates all of it into one report. Modeled directly on
# submit_evaluate_all_baselines_pretrained.sh (which this supersedes as the
# official protocol, and which still works standalone against the old
# official test sets): reuses this repo's existing per-model submission
# scripts as subprocesses rather than duplicating their singularity/torchrun
# invocations (each one's own #SBATCH header is inert when run via `bash`
# instead of `sbatch`).
#
# Each model has its own prerequisites (a separate Singularity image plus
# staged checkpoint(s) for the four baselines -- see that script's own header
# comment; scripts/data_prep/resample_locusbench.py must have already been
# run for Grounder/finetuned-VoxTell's resampled grid, and
# submit_precompute_locusbench_{ed,onc}_embeddings.sh for Grounder's text
# cache, if the checkpoint was trained with data.embedding_cache set). If a
# model isn't set up yet, or its run errors out for any other reason (OOM,
# bad path, etc.), this script logs it and moves on to the next model rather
# than aborting the whole job.
#
# Usage: sbatch submit_evaluate_all_locusbench.sh <grounder_checkpoint> <voxtell_finetuned_checkpoint> [output_dir]
#   e.g. sbatch submit_evaluate_all_locusbench.sh \
#            runs/default/checkpoints/best.pt runs/voxtell_finetune/checkpoints/best.pt \
#            outputs/eval/all_locusbench
#   -> writes outputs/eval/all_locusbench/<model>/{onc,ed}/results.json
#      and    outputs/eval/all_locusbench/summary.txt
#
# If you have six GPUs available, it's faster to `sbatch` the individual
# submit_evaluate_locusbench_*.sh scripts separately instead of this one,
# then run report_baseline_results.py once they've all finished.

GROUNDER_CHECKPOINT=${1:?Usage: sbatch submit_evaluate_all_locusbench.sh <grounder_checkpoint> <voxtell_finetuned_checkpoint> [output_dir]}
VOXTELL_FINETUNED_CHECKPOINT=${2:?Usage: sbatch submit_evaluate_all_locusbench.sh <grounder_checkpoint> <voxtell_finetuned_checkpoint> [output_dir]}
OUTPUT_DIR=${3:-outputs/eval/all_locusbench}

REPO=$HOME/grounder
SIF=$REPO/grounder.sif   # only used for the final report step below
SCRIPT_DIR=$REPO/submission_scripts/evaluation

mkdir -p $REPO/logs

declare -A STATUS

run_model() {
  local name=$1; shift
  echo ""
  echo "########## $name ##########"
  if bash "$@"; then
    STATUS[$name]="completed (see report below for per-split pass/fail)"
  else
    STATUS[$name]="FAILED (exit $?) -- skipped, continuing with remaining models"
    echo "!!! $name FAILED -- continuing with remaining models !!!" >&2
  fi
}

run_model grounder_ed          "$SCRIPT_DIR/submit_evaluate_locusbench_grounder_ed.sh" "$GROUNDER_CHECKPOINT" "$OUTPUT_DIR/grounder/ed"
run_model grounder_onc         "$SCRIPT_DIR/submit_evaluate_locusbench_grounder_onc.sh" "$GROUNDER_CHECKPOINT" "$OUTPUT_DIR/grounder/onc"
run_model voxtell_finetuned_ed  "$SCRIPT_DIR/submit_evaluate_locusbench_voxtell_finetuned_ed.sh" "$VOXTELL_FINETUNED_CHECKPOINT" "$OUTPUT_DIR/voxtell_finetuned/ed"
run_model voxtell_finetuned_onc "$SCRIPT_DIR/submit_evaluate_locusbench_voxtell_finetuned_onc.sh" "$VOXTELL_FINETUNED_CHECKPOINT" "$OUTPUT_DIR/voxtell_finetuned/onc"
run_model voxtell_pretrained "$SCRIPT_DIR/submit_evaluate_locusbench_voxtell_pretrained.sh" "$OUTPUT_DIR/voxtell_pretrained"
run_model sat                "$SCRIPT_DIR/submit_evaluate_locusbench_sat.sh" "$OUTPUT_DIR/sat"
run_model segvol             "$SCRIPT_DIR/submit_evaluate_locusbench_segvol.sh" "$OUTPUT_DIR/segvol"
run_model biomedparse        "$SCRIPT_DIR/submit_evaluate_locusbench_biomedparse.sh" "$OUTPUT_DIR/biomedparse"

echo ""
echo "########## Job status per model ##########"
for name in grounder_ed grounder_onc voxtell_finetuned_ed voxtell_finetuned_onc voxtell_pretrained sat segvol biomedparse; do
  echo "$name: ${STATUS[$name]}"
done

echo ""
echo "########## Results report ##########"
singularity run --nv \
  --pwd /workspace \
  --bind $REPO:/workspace \
  $SIF \
  python scripts/evaluation/report_baseline_results.py "$OUTPUT_DIR" --by-finding
