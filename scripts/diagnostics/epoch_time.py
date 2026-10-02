#!/usr/bin/env python3
"""
Estimates average epoch wall-clock time from a run's metrics.jsonl.

Grounder runs (training/trainer.py's Trainer._log_json) log a "train_step"
record every training.log_every steps, in addition to one "epoch" record per
epoch. For those, per-epoch time is estimated from the "train_step" records'
timestamps (first vs. last logged step within that epoch, extrapolated to the
epoch's last step), NOT from gaps between "epoch" records -- a run chained
across multiple SLURM jobs (e.g. submit_h200_4gpu.sh, which resubmits itself
once per epoch) can have arbitrary queue-wait time between one epoch's "epoch"
record and the next job's first record, which would otherwise swamp the
estimate.

VoxTell runs (finetune_voxtell.py's log_metrics) only ever log one record per
epoch (no intermediate step-level logging), so there's nothing to extrapolate
from -- this script falls back to gaps between consecutive epoch records for
those. finetune_voxtell.py's own submission scripts run continuously within a
single job (no per-epoch resubmission), so that gap isn't queue-wait-inflated
the way Grounder's would be.

Usage
-----
    python epoch_time.py runs/h200_4gpu/<run_name>/metrics.jsonl
    python epoch_time.py runs/voxtell_scratch_h200_4gpu_30pct/metrics.jsonl
"""
import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics_jsonl", type=Path)
    return parser.parse_args()


def _from_train_steps(records: list[dict]) -> list[float]:
    """One estimate per epoch: (last - first) logged train_step timestamp, divided by
    the step distance between them, extrapolated to the epoch's last logged step
    (a proxy for its total step count -- logging stops within log_every steps of the
    epoch's actual end)."""
    by_epoch: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_epoch[r["epoch"]].append(r)

    estimates = []
    for epoch, recs in by_epoch.items():
        recs.sort(key=lambda r: r["step"])
        first, last = recs[0], recs[-1]
        step_delta = last["step"] - first["step"]
        if step_delta <= 0:
            continue
        per_step = (last["time"] - first["time"]).total_seconds() / step_delta
        estimates.append(per_step * last["step"])
    return estimates


def _from_epoch_gaps(records: list[dict]) -> list[float]:
    ordered = [r["time"] for r in sorted(records, key=lambda r: r["epoch"])]
    return [(b - a).total_seconds() for a, b in zip(ordered, ordered[1:])]


def main() -> None:
    args = parse_args()

    train_step_records = []
    epoch_records = []
    with open(args.metrics_jsonl) as f:
        for line in f:
            record = json.loads(line)
            record["time"] = datetime.fromisoformat(record["time"])
            if record.get("phase") == "train_step":
                train_step_records.append(record)
            elif record.get("phase") == "epoch" or ("epoch" in record and "phase" not in record):
                epoch_records.append(record)

    if train_step_records:
        print(f"Using train_step records ({len(train_step_records)} found) -- extrapolated per-epoch estimate.")
        estimates = _from_train_steps(train_step_records)
    else:
        print(f"No train_step records found -- falling back to gaps between {len(epoch_records)} epoch records.")
        estimates = _from_epoch_gaps(epoch_records)

    if len(estimates) < 1:
        raise SystemExit("Not enough data to estimate epoch time.")

    print(f"{len(estimates)} epoch estimate(s)")
    print(f"mean:   {statistics.mean(estimates):.1f}s")
    print(f"median: {statistics.median(estimates):.1f}s")
    print(f"min:    {min(estimates):.1f}s")
    print(f"max:    {max(estimates):.1f}s")


if __name__ == "__main__":
    main()
