#!/usr/bin/env python3
"""
Histogram a specific manifest sample's resolved image file to find what's
producing a suspicious, exactly-repeated median (e.g. p50=0.538, seen
identically across multiple unrelated accessions in
visualize_data_orientation.py's percentile logging) -- a real, independent CT
scan's median should not match another scan's to 3 decimal places, so this
points at a constant value dominating over half the volume rather than
gradual interpolation drift.

Loads RAW (pre-_apply_windows) HU values directly via load_nifti_canonical,
so the printed bins are in HU, not normalized [-1, 1] space.

Usage
-----
    python check_manifest_sample_histogram.py --manifest official_splits/all_data_train.json \
        --mask-id CASE0000000/mask_7_39_1.nii.gz \
        --image-dir /path/to/data/inhouse_abdominal_ct/nifti_resampled \
        --mask-dir /path/to/data/inhouse_abdominal_ct/labels_resampled
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import load_nifti_canonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mask-id", required=True, help="The manifest entry's \"mask\" field value")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=1000.0)
    args = parser.parse_args()

    with open(args.manifest) as f:
        samples = json.load(f)
    matches = [s for s in samples if s["mask"] == args.mask_id]
    if not matches:
        print(f"No manifest entry with mask == {args.mask_id!r}")
        return
    sample = matches[0]

    image_path = Path(args.image_dir) / sample["image"]
    mask_path = Path(args.mask_dir) / sample["mask"]
    print(f"manifest entry: {sample}")
    print(f"resolved image path: {image_path}  (exists={image_path.exists()})")
    print(f"resolved mask path:  {mask_path}  (exists={mask_path.exists()})")

    image = load_nifti_canonical(str(image_path))
    print(f"\nimage shape={image.shape}  dtype={image.dtype}")
    print(f"min={image.min():.1f}  max={image.max():.1f}  mean={image.mean():.1f}  median={np.median(image):.1f}")

    # Coarse histogram in 50 HU bins across the full clipped range, to spot a spike
    bin_edges = np.arange(-1050, 1100, 50)
    counts, edges = np.histogram(image, bins=bin_edges)
    total = image.size
    print("\nHU histogram (50 HU bins, showing bins with >1% of voxels):")
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        frac = c / total
        if frac > 0.01:
            bar = "#" * int(frac * 100)
            print(f"  [{lo:6.0f}, {hi:6.0f}): {frac:6.2%}  {bar}")

    # Exact-value mode: is there a single value repeated so often it dominates?
    flat = image.ravel()
    sample_idx = np.random.default_rng(0).choice(flat.size, size=min(2_000_000, flat.size), replace=False)
    counter = Counter(np.round(flat[sample_idx], 1).tolist())
    most_common = counter.most_common(5)
    print("\nMost common exact HU values (rounded to 0.1, sampled up to 2M voxels):")
    for val, cnt in most_common:
        print(f"  {val:8.1f} HU: {cnt} occurrences ({cnt/len(sample_idx):.2%} of sampled voxels)")

    # ---- Now replicate GrounderDataset._apply_windows + _pad_to_divisible + trim
    # (copied inline, not instantiating GrounderDataset, to avoid needing a
    # tokenizer/embedding_cache) on this SAME loaded array, to see whether the
    # discrepancy against visualize_data_orientation.py's logged percentiles for
    # this exact file is introduced by that processing chain rather than by
    # loading different data across separate script runs.
    hu_min, hu_max = args.hu_min, args.hu_max
    clipped = np.clip(image, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min) * 2.0 - 1.0  # (D, H, W)

    divisor = 16
    D, H, W = normalized.shape
    pad_D = (divisor - D % divisor) % divisor
    pad_H = (divisor - H % divisor) % divisor
    pad_W = (divisor - W % divisor) % divisor
    t = torch.from_numpy(normalized).float().unsqueeze(0)  # (1, D, H, W), matches _apply_windows' (C,D,H,W)
    padding = (0, pad_W, 0, pad_H, 0, pad_D)
    padded = F.pad(t, padding, mode="constant", value=-1.0)
    print(f"\n_pad_to_divisible: ({D},{H},{W}) -> {tuple(padded.shape[1:])}  "
          f"(pad_D={pad_D}, pad_H={pad_H}, pad_W={pad_W})")

    # Trim back exactly like visualize_data_orientation.py's _trim_volume does
    trimmed = padded[0].numpy()
    Dt = trimmed.shape[0] - pad_D if pad_D > 0 else trimmed.shape[0]
    Ht = trimmed.shape[1] - pad_H if pad_H > 0 else trimmed.shape[1]
    Wt = trimmed.shape[2] - pad_W if pad_W > 0 else trimmed.shape[2]
    trimmed = trimmed[:Dt, :Ht, :Wt]

    print(f"trimmed shape={trimmed.shape}  matches raw shape={trimmed.shape == image.shape}")
    print(f"trimmed == normalized (pre-pad) everywhere: {np.array_equal(trimmed, normalized)}")

    raw_pct = np.percentile(image, [1, 5, 50, 95, 99])
    norm_pct = np.percentile(trimmed, [1, 5, 50, 95, 99])
    expected_norm_pct = (np.clip(raw_pct, hu_min, hu_max) - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
    print("\nPercentile comparison (raw HU vs trimmed-normalized vs expected-from-raw):")
    for p, r, n, e in zip([1, 5, 50, 95, 99], raw_pct, norm_pct, expected_norm_pct):
        flag = "" if abs(n - e) < 1e-6 else "  <-- MISMATCH"
        print(f"  p{p:<3} raw={r:8.1f} HU   trimmed_normalized={n:7.3f}   expected={e:7.3f}{flag}")


if __name__ == "__main__":
    main()
