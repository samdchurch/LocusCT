#!/usr/bin/env python3
"""
Finds NIfTI files under --nifti-dir that aren't referenced by any of
--manifests' "image" entries (default: the train/val/test splits
configs/default.yaml uses). Manifest paths are relative to --nifti-dir,
the same convention data/dataset.py resolves them against.

Usage
-----
    python check_unused_nifti_files.py
    python check_unused_nifti_files.py --nifti-dir /path/to/nifti
    python check_unused_nifti_files.py --manifests official_splits/all_data_train.json official_splits/curated_ed_onc_val_data.json
    python check_unused_nifti_files.py --output unused_nifti_files.json

    # Check masks against labels/ instead of images against nifti/:
    python check_unused_nifti_files.py --nifti-dir /path/to/labels --field mask
"""

import argparse
import json
from pathlib import Path

DEFAULT_NIFTI_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"
DEFAULT_MANIFESTS = [
    "official_splits/all_data_train.json",
    "official_splits/curated_ed_onc_val_data.json",
    "official_splits/all_test_data.json",
]


def referenced_paths(manifest_paths: list[str], field: str) -> set[str]:
    referenced = set()
    for mp in manifest_paths:
        with open(mp) as f:
            data = json.load(f)
        for entry in data:
            referenced.add(entry[field])
    return referenced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nifti-dir", default=DEFAULT_NIFTI_DIR,
                         help=f"Directory of <accession>/*.nii.gz files to check (default: {DEFAULT_NIFTI_DIR})")
    parser.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS,
                         help=f"Manifest JSON files to check against (default: {DEFAULT_MANIFESTS})")
    parser.add_argument("--field", default="image", choices=["image", "mask"],
                         help="Manifest key to compare file paths against (default: image)")
    parser.add_argument("--output", default=None,
                         help="Optional path to write the unreferenced/missing file lists as JSON")
    args = parser.parse_args()

    nifti_dir = Path(args.nifti_dir)
    on_disk = {
        str(p.relative_to(nifti_dir)).replace("\\", "/")
        for p in nifti_dir.glob("*/*.nii.gz")
    }

    referenced = referenced_paths(args.manifests, field=args.field)

    unreferenced = sorted(on_disk - referenced)
    missing_from_disk = sorted(referenced - on_disk)

    print(f"On disk under {nifti_dir}: {len(on_disk)} file(s)")
    print(f"Referenced across {len(args.manifests)} manifest(s) (field={args.field}): {len(referenced)} unique path(s)")
    print(f"On disk but NOT in any split: {len(unreferenced)}")
    if missing_from_disk:
        print(f"WARNING: referenced in a split but missing from disk: {len(missing_from_disk)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"unreferenced": unreferenced, "missing_from_disk": missing_from_disk}, f, indent=2)
        print(f"Wrote details to {args.output}")


if __name__ == "__main__":
    main()
