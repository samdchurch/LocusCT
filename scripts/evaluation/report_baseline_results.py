#!/usr/bin/env python3
"""Aggregate results.json files from the per-model pretrained/finetuned eval
runs x (ONC, ED) into one summary table. Tolerant of missing or malformed
files -- e.g. because one model's eval run errored out and was skipped by
submit_evaluate_all_locusbench.py (or the old
submit_evaluate_all_baselines_pretrained.sh, which this still works with).

Usage: python scripts/evaluation/report_baseline_results.py <output_dir> [--by-finding]
  where <output_dir> contains <output_dir>/<model>/<split>/results.json
  for model in {grounder, voxtell_pretrained, voxtell_finetuned, sat, segvol,
  biomedparse}, split in {onc, ed}.
"""
import argparse
import json
from pathlib import Path

MODELS = ["grounder", "voxtell_pretrained", "voxtell_finetuned", "sat", "segvol", "biomedparse"]
SPLITS = ["onc", "ed"]


def format_stats(stats: dict) -> str:
    return (f"Dice {stats['dice_mean']:.4f} +/- {stats['dice_std']:.4f}   "
            f"Hit {stats['hit_rate']:.4f}   N={stats['n_samples']}")


def load_results(path: Path) -> tuple[dict | None, str | None]:
    """Returns (parsed_json, error_message) -- exactly one is None."""
    if not path.exists():
        return None, "MISSING (no results.json -- run likely failed before writing output)"
    try:
        return json.loads(path.read_text()), None
    except Exception as e:
        return None, f"ERROR reading {path.name}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--by-finding", action="store_true",
                     help="Also print each model/split's by_category breakdown "
                          "(every eval script populates this, keyed by the manifest's "
                          "'finding' field -- 'Unknown' where finding is null)")
    args = ap.parse_args()

    header = f"{'model':<18} {'split':<6} results"
    lines = [header, "-" * 80]
    for model in MODELS:
        for split in SPLITS:
            data, err = load_results(args.output_dir / model / split / "results.json")
            if err:
                lines.append(f"{model:<18} {split:<6} {err}")
                continue
            lines.append(f"{model:<18} {split:<6} {format_stats(data['overall'])}")
            if args.by_finding:
                by_category = data.get("by_category", {})
                for cat, stats in sorted(by_category.items()):
                    lines.append(f"{'':<18} {'':<6}   {cat:<20} {format_stats(stats)}")
    report = "\n".join(lines)

    print(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
