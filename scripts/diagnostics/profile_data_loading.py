#!/usr/bin/env python3
"""
Profile how long it takes to load each sample through the full data pipeline
and transfer it to GPU memory.

Measures three phases per batch:
  fetch_s      -- time for DataLoader to deliver a batch (disk I/O, gzip
                  decompress, CPU preprocessing, collation, pin_memory)
  transfer_s   -- time for the H2D copy to complete (image + mask + text)
  total_s      -- fetch_s + transfer_s

Reports per-sample mean, std, p50/p95/p99, and overall throughput.
Outlier batches (>mean+3*std total time) are listed with their sample IDs
so you can spot consistently slow files.

Usage
-----
    python profile_data_loading.py --config configs/default.yaml
    python profile_data_loading.py --config configs/default.yaml \\
        --split val --n_batches 200 --warmup 10
    python profile_data_loading.py --config configs/default.yaml \\
        --override data.train_manifest=official_splits/benchmark_1000_gz.json \\
                   data.embedding_cache=""
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import build_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", metavar="KEY=VALUE",
                         help="Dot-notation config overrides")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n_batches", type=int, default=100,
                         help="Number of batches to time after warmup")
    parser.add_argument("--warmup", type=int, default=5,
                         help="Batches to discard before recording (pipeline warm-up)")
    parser.add_argument("--out", default=None,
                         help="Optional path to write per-batch JSON results")
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        key, _, raw_val = kv.partition("=")
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw_val)
    return cfg


def stats(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "mean":  float(a.mean()),
        "std":   float(a.std()),
        "min":   float(a.min()),
        "p50":   float(np.percentile(a, 50)),
        "p95":   float(np.percentile(a, 95)),
        "p99":   float(np.percentile(a, 99)),
        "max":   float(a.max()),
    }


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    manifest_key = f"{args.split}_manifest"
    loader = build_dataloader(cfg["data"][manifest_key], cfg, split=args.split)
    batch_size = cfg["training"]["batch_size"]
    log.info(f"Split: {args.split}  |  dataset size: {len(loader.dataset)}  "
              f"|  batch_size: {batch_size}  |  num_workers: {loader.num_workers}")
    log.info(f"Warmup batches: {args.warmup}  |  Timing batches: {args.n_batches}")

    fetch_times, transfer_times, sample_ids = [], [], []

    loader_iter = iter(loader)
    total_needed = args.warmup + args.n_batches

    for i in range(total_needed):
        # --- fetch (CPU pipeline) ---
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            log.warning("DataLoader exhausted after %d batches; stopping early", i)
            break
        t1 = time.perf_counter()

        # --- H2D transfer ---
        image = batch["image"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)
        if "text_feats" in batch:
            batch["text_feats"].to(device, non_blocking=True)
            batch["text_padding_mask"].to(device, non_blocking=True)
        else:
            batch["input_ids"].to(device, non_blocking=True)
            batch["attention_mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        if i < args.warmup:
            log.info(f"[warmup {i+1}/{args.warmup}]  "
                      f"fetch={t1-t0:.3f}s  transfer={t2-t1:.3f}s  "
                      f"image={list(image.shape)}")
            continue

        fetch_times.append(t1 - t0)
        transfer_times.append(t2 - t1)
        sample_ids.append(batch["id"] if isinstance(batch["id"], list) else [batch["id"]])

        if (i - args.warmup + 1) % 20 == 0:
            n_done = i - args.warmup + 1
            elapsed = sum(fetch_times) + sum(transfer_times)
            log.info(f"  [{n_done}/{args.n_batches}]  "
                      f"fetch={fetch_times[-1]:.3f}s  transfer={transfer_times[-1]:.3f}s  "
                      f"throughput={n_done * batch_size / elapsed:.2f} samples/s")

    if not fetch_times:
        log.error("No timing data collected — too few batches in dataset?")
        sys.exit(1)

    total_times = [f + t for f, t in zip(fetch_times, transfer_times)]
    n_samples = len(fetch_times) * batch_size
    wall = sum(total_times)

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS  (per-sample, seconds)")
    log.info("=" * 60)

    for label, values in [("fetch    ", fetch_times), ("transfer ", transfer_times), ("total    ", total_times)]:
        s = stats([v / batch_size for v in values])
        log.info(f"  {label}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"p50={s['p50']:.4f}  p95={s['p95']:.4f}  p99={s['p99']:.4f}  "
                  f"max={s['max']:.4f}")

    log.info(f"  throughput  {n_samples / wall:.2f} samples/s  "
              f"({n_samples} samples in {wall:.1f}s)")

    # Flag outlier batches (total time > mean + 3*std) to identify slow files
    mean_t = np.mean(total_times)
    std_t  = np.std(total_times)
    threshold = mean_t + 3 * std_t
    outliers = [(i, t, ids) for i, (t, ids) in enumerate(zip(total_times, sample_ids)) if t > threshold]
    if outliers:
        log.info(f"\n  Outlier batches (>{threshold:.3f}s):")
        for i, t, ids in outliers:
            log.info(f"    batch {i:4d}  total={t:.3f}s  ids={ids}")

    if args.out:
        result = {
            "config": args.config,
            "split": args.split,
            "n_batches": len(fetch_times),
            "batch_size": batch_size,
            "device": str(device),
            "fetch_s":    stats([v / batch_size for v in fetch_times]),
            "transfer_s": stats([v / batch_size for v in transfer_times]),
            "total_s":    stats([v / batch_size for v in total_times]),
            "throughput_samples_per_s": n_samples / wall,
            "per_batch": [
                {"batch": i, "ids": ids, "fetch_s": f, "transfer_s": t}
                for i, (ids, f, t) in enumerate(zip(sample_ids, fetch_times, transfer_times))
            ],
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        log.info(f"\nPer-batch results written to {args.out}")


if __name__ == "__main__":
    main()
