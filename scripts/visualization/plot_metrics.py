"""
Plot training curves from metrics.jsonl, or eval distributions from evaluate.py JSON output.

Usage:
    # Training curves (one or more runs to compare):
    python plot_metrics.py runs/h200/metrics.jsonl [runs/l40s/metrics.jsonl ...] [--labels h200 l40s]

    # Eval metric distributions:
    python plot_metrics.py eval_val.json [eval_test.json ...] [--labels val test]

    # Custom output path:
    python plot_metrics.py ... --output my_plot.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default=None)
    parser.add_argument("--labels", nargs="*", default=None)
    return parser.parse_args()


def _ax_style(ax: plt.Axes) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


def _legend(ax: plt.Axes, **kwargs) -> None:
    ax.legend(fontsize=8, facecolor="black", labelcolor="white", edgecolor="#444444", **kwargs)


def _smooth(values: list[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return np.arange(len(arr)), arr
    smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
    xs = np.arange(window // 2, window // 2 + len(smoothed))
    return xs, smoothed


def _parse_jsonl(path: str) -> tuple[list[dict], list[dict]]:
    steps, epochs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            phase = entry.get("phase", "")
            if phase == "train_step":
                steps.append(entry)
            elif phase == "epoch":
                epochs.append(entry)
    return steps, epochs


# ── Training curves ──────────────────────────────────────────────────────────

def _epoch_lines(ax: plt.Axes, runs: list, labels: list[str], colors,
                 train_key: str, val_key: str) -> None:
    for i, ((_, epochs), label) in enumerate(zip(runs, labels)):
        ep = [e["epoch"] for e in epochs]
        ax.plot(ep, [e.get(train_key, float("nan")) for e in epochs],
                color=colors[i], lw=2, label=f"{label} train")
        ax.plot(ep, [e.get(val_key, float("nan")) for e in epochs],
                color=colors[i], lw=2, ls="--", label=f"{label} val")


def _plot_training(runs: list, labels: list[str], colors, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("black")
    for ax in axes.flat:
        _ax_style(ax)

    # Loss
    ax = axes[0, 0]
    _epoch_lines(ax, runs, labels, colors, "train_loss", "val_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
    _legend(ax)

    # Dice
    ax = axes[0, 1]
    _epoch_lines(ax, runs, labels, colors, "train_dice", "val_dice")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Dice"); ax.set_title("Dice Score")
    ax.set_ylim(bottom=0)
    _legend(ax)

    # Hit Rate
    ax = axes[0, 2]
    _epoch_lines(ax, runs, labels, colors, "train_hit_rate", "val_hit_rate")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hit Rate"); ax.set_title("Hit Rate (Dice ≥ 0.1)")
    ax.set_ylim(0, 1)
    _legend(ax)

    # IoU
    ax = axes[1, 0]
    _epoch_lines(ax, runs, labels, colors, "train_iou", "val_iou")
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU"); ax.set_title("IoU")
    ax.set_ylim(bottom=0)
    _legend(ax)

    # Learning Rate
    ax = axes[1, 1]
    for i, ((_, epochs), label) in enumerate(zip(runs, labels)):
        ep = [e["epoch"] for e in epochs]
        lrs = [e.get("lr", float("nan")) for e in epochs]
        ax.plot(ep, lrs, color=colors[i], lw=2, label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning Rate")
    _legend(ax)

    # Step-level loss (smoothed), epoch boundaries as vertical lines
    smooth_w = 30
    ax = axes[1, 2]
    for i, ((steps, _), label) in enumerate(zip(runs, labels)):
        if not steps:
            continue
        losses = [s["loss"] for s in steps]
        ax.plot(range(len(losses)), losses, alpha=0.1, color=colors[i], lw=0.5)
        xs_s, smoothed = _smooth(losses, smooth_w)
        ax.plot(xs_s, smoothed, color=colors[i], lw=1.5, label=label)
        prev_ep = steps[0]["epoch"]
        for j, s in enumerate(steps):
            if s["epoch"] != prev_ep:
                ax.axvline(j, color="#444444", lw=0.5, alpha=0.6)
                prev_ep = s["epoch"]
    ax.set_xlabel("Step (sequential)"); ax.set_ylabel("Loss")
    ax.set_title(f"Step Loss (smooth={smooth_w}, │ = epoch boundary)")
    _legend(ax)

    parts = []
    for (_, epochs), label in zip(runs, labels):
        if epochs:
            last = epochs[-1]
            parts.append(
                f"{label} ep{last['epoch']}: "
                f"train_dice={last.get('train_dice', float('nan')):.3f}  "
                f"val_dice={last.get('val_dice', float('nan')):.3f}  "
                f"val_hit={last.get('val_hit_rate', float('nan')):.3f}"
            )
    fig.suptitle("\n".join(parts), color="white", fontsize=9, y=1.02)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Saved → {out_path}")


# ── Eval distributions ───────────────────────────────────────────────────────

def _plot_eval(runs: list[dict], labels: list[str], colors, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("black")
    for ax in axes.flat:
        _ax_style(ax)

    # Dice histogram
    ax = axes[0, 0]
    for i, (run, label) in enumerate(zip(runs, labels)):
        dices = [s["dice"] for s in run["per_sample"]]
        ax.hist(dices, bins=40, range=(0, 1), alpha=0.6, color=colors[i],
                label=f"{label}  μ={run['dice_mean']:.3f} ±{run['dice_std']:.3f}")
        ax.axvline(run["dice_mean"], color=colors[i], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Dice"); ax.set_ylabel("Count"); ax.set_title("Dice Distribution")
    _legend(ax)

    # IoU histogram
    ax = axes[0, 1]
    for i, (run, label) in enumerate(zip(runs, labels)):
        ious = [s["iou"] for s in run["per_sample"]]
        ax.hist(ious, bins=40, range=(0, 1), alpha=0.6, color=colors[i],
                label=f"{label}  μ={run['iou_mean']:.3f} ±{run['iou_std']:.3f}")
        ax.axvline(run["iou_mean"], color=colors[i], linestyle="--", linewidth=1.5)
    ax.set_xlabel("IoU"); ax.set_ylabel("Count"); ax.set_title("IoU Distribution")
    _legend(ax)

    # Precision–Recall scatter
    ax = axes[0, 2]
    for i, (run, label) in enumerate(zip(runs, labels)):
        precs = [s["precision"] for s in run["per_sample"]]
        recs  = [s["recall"]    for s in run["per_sample"]]
        ax.scatter(recs, precs, alpha=0.25, s=6, color=colors[i])
        ax.scatter([run["recall_mean"]], [run["precision_mean"]],
                   marker="*", s=220, color=colors[i], edgecolors="white",
                   linewidths=0.5, zorder=5, label=label)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Precision vs Recall  (★ = mean)")
    _legend(ax)

    # Dice CDF
    ax = axes[1, 0]
    thresholds = np.linspace(0, 1, 300)
    for i, (run, label) in enumerate(zip(runs, labels)):
        dices = np.array([s["dice"] for s in run["per_sample"]])
        cdf = [(dices >= t).mean() for t in thresholds]
        ax.plot(thresholds, cdf, color=colors[i], linewidth=2, label=label)
    ax.axvline(0.1, color="#888888", linestyle=":", linewidth=1, label="0.1 (hit rate)")
    ax.axvline(0.5, color="#666666", linestyle=":", linewidth=1, label="0.5")
    ax.set_xlabel("Dice threshold"); ax.set_ylabel("Fraction ≥ threshold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title("Dice CDF")
    _legend(ax)

    # Box plot
    ax = axes[1, 1]
    metric_keys   = ["dice", "iou", "precision", "recall"]
    metric_labels = ["Dice", "IoU", "Prec", "Rec"]
    n_runs = len(runs)
    bar_w = 0.7 / n_runs
    for i, (run, label) in enumerate(zip(runs, labels)):
        data = [[s[k] for s in run["per_sample"]] for k in metric_keys]
        positions = [j + (i - n_runs / 2 + 0.5) * bar_w for j in range(len(metric_keys))]
        bp = ax.boxplot(data, positions=positions, widths=bar_w * 0.85,
                        patch_artist=True, manage_ticks=False,
                        medianprops=dict(color="white", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[i]); patch.set_alpha(0.7)
        for element in ["whiskers", "caps", "fliers"]:
            for line in bp[element]:
                line.set_color(colors[i])
    ax.set_xticks(range(len(metric_keys))); ax.set_xticklabels(metric_labels, color="white")
    ax.set_ylim(-0.05, 1.05); ax.set_title("Metrics Box Plot"); ax.set_ylabel("Score")
    handles = [mpatches.Patch(facecolor=colors[i], alpha=0.7, label=lbl)
               for i, lbl in enumerate(labels)]
    ax.legend(handles=handles, fontsize=8, facecolor="black", labelcolor="white", edgecolor="#444444")

    # Sorted Dice percentile
    ax = axes[1, 2]
    for i, (run, label) in enumerate(zip(runs, labels)):
        dices = sorted([s["dice"] for s in run["per_sample"]])
        x = np.linspace(0, 100, len(dices))
        ax.plot(x, dices, color=colors[i], linewidth=1.5, label=label)
        ax.fill_between(x, dices, alpha=0.15, color=colors[i])
    ax.set_xlabel("Percentile"); ax.set_ylabel("Dice")
    ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.set_title("Sorted Dice per Sample")
    _legend(ax)

    parts = []
    for run, label in zip(runs, labels):
        parts.append(
            f"{label}: Dice {run['dice_mean']:.3f}±{run['dice_std']:.3f}  "
            f"IoU {run['iou_mean']:.3f}  "
            f"P {run['precision_mean']:.3f}  R {run['recall_mean']:.3f}  "
            f"Hit@0.1 {run['hit_rate']:.3f}  n={run['n_samples']}"
        )
    fig.suptitle("\n".join(parts), color="white", fontsize=9, y=1.02)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    labels = args.labels or [Path(p).stem for p in args.results]
    colors = plt.cm.tab10.colors
    out_path = Path(args.output) if args.output else Path(args.results[0]).with_suffix(".png")

    if Path(args.results[0]).suffix == ".jsonl":
        runs = [_parse_jsonl(p) for p in args.results]
        _plot_training(runs, labels, colors, out_path)
    else:
        runs = []
        for p in args.results:
            with open(p) as f:
                runs.append(json.load(f))
        _plot_eval(runs, labels, colors, out_path)


if __name__ == "__main__":
    main()
