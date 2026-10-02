#!/usr/bin/env python3
"""
Trains and evaluates every ablation in configs/ablation_studies.yaml, one at a
time. Each ablation trains on a fraction of the training manifest for a small
number of epochs (see QUICK-RUN BUDGET below), then evaluates the resulting
best.pt on --split. Writes a runs/ablations/<name>/DONE marker only once both
steps succeed, and skips any ablation whose marker already exists -- so a
killed/requeued job resumes without redoing finished ablations.

Usage:
    python run_ablations.py --config configs/default.yaml \
        --override model.text_encoder_name=/path/to/Qwen3-Embedding-8B
"""
import argparse
import copy
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from train import apply_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ABLATIONS_FILE = "configs/ablation_studies.yaml"
OUT_ROOT = Path("runs/ablations")

# --- Quick-run budget ---
# 10% of the training manifest, 10 epochs. This is a screening pass, not a
# final result: at 10% data x 10 epochs it sees ~1% of the sample-passes a
# full run (100% data x 100 epochs) does, so treat rankings as directional --
# confirm whatever wins here with a longer run before drawing real conclusions.
# warmup_epochs is scaled down to 1 (matching default.yaml's 10/100 = 10%
# warmup ratio) rather than left at 10 -- otherwise, with only 10 total
# epochs, the LR would warm up for the entire run and never cosine-decay, and
# Trainer's dice_weight ramp (tied to scheduler.warmup_epochs) would never
# finish ramping or hold at its endpoint, breaking the loss_* ablations'
# premise of comparing held endpoint values.
DATA_FRACTION = 0.10
NUM_EPOCHS = 10
WARMUP_EPOCHS = 1


def _manifest_len(entry: str | dict) -> int:
    path = entry["manifest"] if isinstance(entry, dict) else entry
    with open(path) as f:
        return len(json.load(f))


def _train_sample_count(cfg: dict) -> int:
    manifest = cfg["data"]["train_manifest"]
    entries = manifest if isinstance(manifest, list) else [manifest]
    return sum(_manifest_len(e) for e in entries)


def _apply_dict_overrides(cfg: dict, overrides: dict) -> dict:
    """Same dot-notation semantics as train.py's apply_overrides, but for
    already-typed values (from parsed YAML) instead of CLI 'KEY=VALUE'
    strings -- avoids round-tripping e.g. an intentional empty-string value
    (data.embedding_cache: "") through yaml.safe_load(""), which parses to
    None instead of "" (functionally equivalent everywhere it's checked via
    `if embedding_cache:`, but there's no reason to rely on that)."""
    for key, val in overrides.items():
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = val
    return cfg


def _run(cmd: list[str]) -> bool:
    logger.info(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def _print_summary(split: str) -> None:
    rows = []
    for eval_path in sorted(OUT_ROOT.glob(f"*/eval_{split}.json")):
        if not (eval_path.parent / "DONE").exists():
            continue
        with open(eval_path) as f:
            summary = json.load(f)
        rows.append((eval_path.parent.name, summary["dice_mean"], summary["dice_std"], summary["iou_mean"]))

    if not rows:
        return
    rows.sort(key=lambda r: r[1], reverse=True)
    logger.info(f"=== Ablation comparison ({split}, best.pt) ===")
    for name, dice_mean, dice_std, iou_mean in rows:
        logger.info(f"  {name:35s} dice={dice_mean:.4f}±{dice_std:.4f}  iou={iou_mean:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="Base config every ablation starts from")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                         help="Ablations are a screening pass -- default val, not test, to keep test held out")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra overrides applied to every ablation before its own, e.g. model.text_encoder_name=/path",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    if args.override:
        base_cfg = apply_overrides(base_cfg, args.override)

    with open(ABLATIONS_FILE) as f:
        ablations = yaml.safe_load(f)["ablations"]

    n_train = _train_sample_count(base_cfg)
    max_samples = round(n_train * DATA_FRACTION)
    logger.info(
        f"Quick-ablation budget: {max_samples}/{n_train} train samples ({DATA_FRACTION:.0%}), "
        f"{NUM_EPOCHS} epochs, {WARMUP_EPOCHS}-epoch warmup"
    )
    quick_overrides = {
        "data.max_samples": max_samples,
        "training.num_epochs": NUM_EPOCHS,
        "scheduler.warmup_epochs": WARMUP_EPOCHS,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}

    for ab in ablations:
        name = ab["name"]
        out_dir = OUT_ROOT / name
        done_marker = out_dir / "DONE"
        if done_marker.exists():
            logger.info(f"[skip] {name} (already done -- {done_marker})")
            results[name] = "skipped (already done)"
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(base_cfg)
        _apply_dict_overrides(cfg, quick_overrides | {"checkpoint.output_dir": str(out_dir)})
        _apply_dict_overrides(cfg, ab["overrides"])

        resolved_config_path = out_dir / "config.yaml"
        with open(resolved_config_path, "w") as f:
            yaml.dump(cfg, f)

        logger.info(f"=== {name} ===")
        if ab.get("hypothesis"):
            logger.info(ab["hypothesis"].strip())

        if not _run([sys.executable, "train.py", "--config", str(resolved_config_path)]):
            logger.error(f"[fail] {name}: training failed, leaving un-marked for retry")
            results[name] = "failed (train)"
            continue

        # train.py nests checkpoints under output_dir / _build_run_name(cfg) (a string
        # encoding channels/batch_size/lr/spatial settings/fusion/max_samples) -- rather
        # than duplicate that naming logic here, just glob for it; out_dir is unique per
        # ablation, so exactly one match is expected.
        checkpoints = list(out_dir.glob("*/checkpoints/best.pt"))
        if len(checkpoints) != 1:
            logger.error(f"[fail] {name}: expected exactly 1 best.pt under {out_dir}, found {len(checkpoints)}")
            results[name] = "failed (checkpoint not found)"
            continue
        checkpoint = checkpoints[0]
        eval_output = out_dir / f"eval_{args.split}.json"
        if not _run([
            sys.executable, "evaluate.py",
            "--config", str(resolved_config_path),
            "--checkpoint", str(checkpoint),
            "--split", args.split,
            "--output", str(eval_output),
        ]):
            logger.error(f"[fail] {name}: evaluation failed, leaving un-marked for retry")
            results[name] = "failed (eval)"
            continue

        done_marker.write_text(json.dumps({"name": name}))
        results[name] = "done"
        logger.info(f"[done] {name}")

    logger.info("Ablation sweep summary:")
    for name, status in results.items():
        logger.info(f"  {name}: {status}")

    _print_summary(args.split)


if __name__ == "__main__":
    main()
