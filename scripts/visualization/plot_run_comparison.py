"""
Compare Dice, Macro Hit Rate, and Loss across runs, with number of optimizer updates
(not epoch) on the x-axis -- lets runs with different epoch lengths (e.g. different
train-set subset fractions, so a fixed epoch count means very different amounts of
actual training) be compared fairly on training compute rather than epoch number.
Dice and Loss each show both train and val lines per run (solid/dashed); Macro Hit
Rate is val-only (no train-side equivalent -- see training/trainer.py::
macro_hit_rate_epoch, only ever computed against held-out ED/ONC validation data).

Reuses plot_metrics.py's own metrics.jsonl parsing/styling helpers (same directory).

Usage:
    python plot_run_comparison.py runs/h200_4gpu_30pct/<run_name> runs/h200_4gpu_10pct/<run_name> \
        --labels 30pct 10pct --output comparison.png

Each positional argument is either a run directory (metrics.jsonl looked up inside it)
or a direct path to a metrics.jsonl file.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_metrics import _ax_style, _legend, _parse_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run directories or metrics.jsonl paths to compare")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output", default="run_comparison.png")
    return parser.parse_args()


def _resolve_metrics_path(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        p = p / "metrics.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"No metrics.jsonl found for {run!r} (looked at {p})")
    return p


def _cumulative_updates(steps: list[dict], epochs: list[dict]) -> list[int]:
    """Cumulative optimizer-update count at the end of each epoch, derived from
    train_step records' own per-epoch "step" counter (resets to 0 each epoch -- see
    training/trainer.py::train_epoch). Only logged every training.log_every_n_steps,
    so this slightly undercounts each epoch's true final step by up to
    log_every_n_steps-1 -- negligible for comparing runs with very different epoch
    lengths, which is the whole point of using update count instead of epoch number.
    """
    max_step_per_epoch: dict[int, int] = {}
    for s in steps:
        ep = s["epoch"]
        max_step_per_epoch[ep] = max(max_step_per_epoch.get(ep, 0), s["step"])
    cumulative = []
    total = 0
    for e in epochs:
        total += max_step_per_epoch.get(e["epoch"], 0)
        cumulative.append(total)
    return cumulative


def _train_val_lines(ax, runs_xy, labels, colors, train_key: str, val_key: str) -> None:
    """Mirrors plot_metrics.py's own _epoch_lines (solid=train, dashed=val, one color
    per run), but plotted against precomputed per-run x-values (update count) instead
    of that function's hardcoded epoch-number x-axis."""
    for i, ((xs, epochs), label) in enumerate(zip(runs_xy, labels)):
        ax.plot(xs, [e.get(train_key, float("nan")) for e in epochs],
                color=colors[i], lw=2, label=f"{label} train")
        ax.plot(xs, [e.get(val_key, float("nan")) for e in epochs],
                color=colors[i], lw=2, ls="--", label=f"{label} val")


def main() -> None:
    args = parse_args()
    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        raise ValueError(f"--labels must have one entry per run ({len(args.runs)} runs, {len(labels)} labels)")

    runs = [_parse_jsonl(str(_resolve_metrics_path(r))) for r in args.runs]
    # (xs, epochs) per run -- xs computed once and reused across all three panels below.
    runs_xy = [(_cumulative_updates(steps, epochs), epochs) for steps, epochs in runs]
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    fig.patch.set_facecolor("black")
    for ax in axes:
        _ax_style(ax)

    ax = axes[0]
    _train_val_lines(ax, runs_xy, labels, colors, "train_dice", "val_dice")
    ax.set_xlabel("Number of updates")
    ax.set_ylabel("Dice")
    ax.set_title("Dice")
    ax.set_ylim(bottom=0)
    _legend(ax)

    ax = axes[1]
    for i, (xs, epochs) in enumerate(runs_xy):
        if not epochs:
            continue
        ys = [e.get("val_macro_hit_rate", float("nan")) for e in epochs]
        ax.plot(xs, ys, color=colors[i], lw=2, label=labels[i])
    ax.set_xlabel("Number of updates")
    ax.set_ylabel("Macro Hit Rate")
    ax.set_title("Macro Hit Rate (val)")
    ax.set_ylim(0, 1)
    _legend(ax)

    ax = axes[2]
    _train_val_lines(ax, runs_xy, labels, colors, "train_loss", "val_loss")
    ax.set_xlabel("Number of updates")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.set_ylim(bottom=0)
    _legend(ax)

    fig.suptitle("Dice, Macro Hit Rate & Loss vs. Training Updates", color="white", fontsize=11, y=1.02)

    plt.tight_layout()
    out_path = Path(args.output)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
