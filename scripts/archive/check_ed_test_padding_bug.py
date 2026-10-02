#!/usr/bin/env python3
"""
Tests whether the ED official test set's images still carry the pre-c010922
fake-water-density-padding bug (crop_pad_to_shape() filling padded regions
with 0.0 HU "water density" instead of -1000 HU "air"), independent of any
model or visualization code.

process_file() in resample_and_crop.py skips accessions whose output already
exists, so the c010922 fix only applies to images resampled after that
commit landed -- any accession resampled earlier (regardless of which
manifest/split references it) still has the old fill value baked in.

For each accession referenced by a manifest, loads its raw resampled image
(no canonicalization, no windowing) and reports the fraction of voxels
exactly equal to 0.0 -- real CT data essentially never lands exactly on
0.0 by chance, so a large fraction is the bug's fingerprint, not noise.

Usage
-----
    python check_ed_test_padding_bug.py --manifest official_splits/ed_official_test_data.json \
        --image-dir /path/to/data/inhouse_abdominal_ct/nifti_resampled
    python check_ed_test_padding_bug.py --manifest official_splits/all_data_train.json \
        --image-dir /path/to/data/inhouse_abdominal_ct/nifti_resampled \
        --n-samples 40
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=None,
                         help="Check only the first N unique accessions (default: all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.manifest) as f:
        samples = json.load(f)

    image_dir = Path(args.image_dir)
    seen_images: dict[str, Path] = {}
    for s in samples:
        seen_images.setdefault(s["image"], image_dir / s["image"])
    images = list(seen_images.values())
    if args.n_samples:
        images = images[: args.n_samples]

    print(f"Checking {len(images)} unique image(s) from {args.manifest}")

    n_ok, n_missing, n_bug, n_clean = 0, 0, 0, 0
    frac_zero_list = []
    frac_air_list = []

    for path in images:
        if not path.exists():
            print(f"MISSING  {path}")
            n_missing += 1
            continue
        data = nib.load(str(path)).get_fdata(dtype=np.float32)
        n_ok += 1
        total = data.size
        frac_zero = float((data == 0.0).sum()) / total
        frac_air = float((data == -1000.0).sum()) / total
        frac_zero_list.append(frac_zero)
        frac_air_list.append(frac_air)
        flag = "BUG " if frac_zero > 0.01 else "clean"
        if frac_zero > 0.01:
            n_bug += 1
        else:
            n_clean += 1
        print(f"{flag:6s} {path.name:35s} shape={data.shape}  "
              f"frac_exact_0.0={frac_zero:.3f}  frac_exact_-1000={frac_air:.3f}")

    print("-" * 70)
    print(f"Checked {n_ok} image(s), {n_missing} missing")
    print(f"  Likely still has pre-fix 0.0-HU padding: {n_bug}/{n_ok}")
    print(f"  Clean (no significant 0.0-HU block):     {n_clean}/{n_ok}")
    if frac_zero_list:
        print(f"  Mean frac_exact_0.0:    {np.mean(frac_zero_list):.3f}")
        print(f"  Mean frac_exact_-1000:  {np.mean(frac_air_list):.3f}")


if __name__ == "__main__":
    main()
