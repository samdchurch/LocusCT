#!/usr/bin/env python3
"""
Concatenates one or more manifest JSON files (each a list of {"image", "mask",
"sentence", "region", "finding"} dicts) into a single output manifest, in the
order given -- e.g. merging official_splits/curated_ed_val_data.json and
official_splits/curated_onc_val_data.json into one combined validation
manifest for data.val_manifest.

Usage:
    python merge_manifests.py \
        --inputs official_splits/curated_ed_val_data.json official_splits/curated_onc_val_data.json \
        --output official_splits/curated_ed_onc_val_data.json
"""
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="Manifest JSON files to concatenate, in order")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = []
    for path in args.inputs:
        with open(path) as f:
            samples = json.load(f)
        merged.extend(samples)
        print(f"{path}: {len(samples)} samples")

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"{out_path}: {len(merged)} samples total")


if __name__ == "__main__":
    main()
