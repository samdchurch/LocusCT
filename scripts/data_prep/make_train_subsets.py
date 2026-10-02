#!/usr/bin/env python3
"""
Creates nested training-manifest subsets from a source manifest -- by default,
1%, 3%, 10%, and 30% of official_splits/all_data_train.json, with each
smaller subset entirely contained in every larger one.

Nesting comes from taking each subset as a prefix of the same seeded random
permutation (subset(f) = permutation[:k], k = round(f * n)) -- every element
of a shorter prefix is by construction an element of every longer one, rather
than independently-sampled subsets that would only partially overlap.

Output manifests are otherwise identical in format to the source (a JSON list
of {"image", "mask", "sentence", "region", "finding"} dicts) -- use them
directly as data.train_manifest in a config.

Usage:
    python make_train_subsets.py
    python make_train_subsets.py --manifest official_splits/all_data_train.json \
        --fractions 0.01 0.03 0.1 0.3 --seed 42
"""
import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="official_splits/all_data_train.json")
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3],
        help="Ascending fractions of the source manifest; each is a prefix of one shared "
             "random permutation, so every smaller subset is contained in every larger one.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Matches the project's default config seed -- unrelated to GrounderDataset's own "
             "runtime max_samples subsampling (data/dataset.py), which is a separate mechanism.",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to the source manifest's own directory")
    args = parser.parse_args()

    if args.fractions != sorted(args.fractions):
        parser.error(f"--fractions must be ascending, got {args.fractions}")
    if not all(0 < f <= 1 for f in args.fractions):
        parser.error(f"--fractions must all be in (0, 1], got {args.fractions}")
    return args


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        samples = json.load(f)
    n = len(samples)

    permutation = list(range(n))
    random.Random(args.seed).shuffle(permutation)

    out_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent
    stem = manifest_path.stem  # e.g. "all_data_train"

    for frac in args.fractions:
        k = round(n * frac)
        # Sorted back to the source manifest's own order, purely for readability
        # (diffing/skimming the output file) -- doesn't affect subset membership.
        indices = sorted(permutation[:k])
        subset = [samples[i] for i in indices]

        pct = round(frac * 100)
        out_path = out_dir / f"{stem}_{pct}pct.json"
        with open(out_path, "w") as f:
            json.dump(subset, f, indent=4)

        print(f"{out_path}: {len(subset)}/{n} samples ({frac:.0%})")


if __name__ == "__main__":
    main()
