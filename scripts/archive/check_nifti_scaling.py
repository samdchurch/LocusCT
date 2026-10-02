#!/usr/bin/env python3
"""
Diagnose why resampled CT intensity values look saturated regardless of display
window choice in visualize_data_orientation.py.

scl_slope/scl_inter were already ruled out (both None on both raw and resampled
files -- no NIfTI-level rescaling is in play; both are stored as plain float64).

Current hypothesis: resample_and_crop.py's resample_volume() calls
scipy.ndimage.zoom(data, zoom_factors, order=3, ...) -- cubic spline
interpolation, which (unlike linear/nearest-neighbor) has no guarantee of
staying within the local min/max of its input neighbors at sharp intensity
edges. Since TARGET_SPACING is usually a downsample relative to native
clinical CT resolution, and zoom()'s prefilter=True only prepares spline
coefficients (not an anti-aliasing low-pass filter matched to the downsample
ratio), this is a known failure mode for scipy.ndimage.zoom at order>1.

A first pass (min/max only) showed the resampled file's range wider than the
raw source's on both ends (e.g. source [-3024, 3071] -> resampled
[-3863, 3782]), consistent with overshoot/ringing rather than a scaling bug.
This adds percentile stats and an out-of-source-range voxel count/fraction to
tell overshoot confined to a few sharp-edge voxels (wouldn't explain
whole-body saturation) apart from a broad perturbation affecting most voxels
(would).

Usage
-----
    python check_nifti_scaling.py --accession CASE0000000 --series 2 \
        --raw-root /path/to/data/inhouse_abdominal_ct/nifti \
        --resampled-root /path/to/data/inhouse_abdominal_ct/nifti_resampled
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

PERCENTILES = [0.1, 1, 5, 50, 95, 99, 99.9]


def describe(path: Path) -> np.ndarray | None:
    if not path.exists():
        print(f"  MISSING: {path}")
        return None

    img = nib.load(str(path))
    header = img.header
    slope, inter = header.get_slope_inter()
    raw_dtype = header.get_data_dtype()
    scaled = img.get_fdata(dtype=np.float32)

    print(f"  {path}")
    print(f"    shape={scaled.shape}  on-disk dtype={raw_dtype}  scl_slope={slope}  scl_inter={inter}")
    print(f"    min/max: [{scaled.min():.1f}, {scaled.max():.1f}]")
    pct = np.percentile(scaled, PERCENTILES)
    pct_str = "  ".join(f"p{p}={v:.1f}" for p, v in zip(PERCENTILES, pct))
    print(f"    percentiles: {pct_str}")
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--series", type=int, required=True)
    parser.add_argument("--raw-root", required=True, help="e.g. .../inhouse_abdominal_ct/nifti")
    parser.add_argument("--resampled-root", required=True, help="e.g. .../inhouse_abdominal_ct/nifti_resampled")
    args = parser.parse_args()

    raw_dir = Path(args.raw_root) / args.accession
    resampled_dir = Path(args.resampled_root) / args.accession

    def find_series_file(d: Path) -> Path | None:
        if not d.is_dir():
            return None
        for f in sorted(d.glob("*.nii.gz")):
            prefix = f.name.split("_")[0]
            if prefix.isdigit() and int(prefix) == args.series:
                return f
        return None

    raw_path = find_series_file(raw_dir)
    resampled_path = find_series_file(resampled_dir)

    print(f"Raw source ({raw_dir}):")
    if raw_path is None:
        print("  no matching series file found")
        raw_data = None
    else:
        raw_data = describe(raw_path)

    print(f"\nResampled output ({resampled_dir}):")
    if resampled_path is None:
        print("  no matching series file found")
        resampled_data = None
    else:
        resampled_data = describe(resampled_path)

    if raw_data is not None and resampled_data is not None:
        lo, hi = raw_data.min(), raw_data.max()
        n_out = int(np.sum((resampled_data < lo) | (resampled_data > hi)))
        frac_out = n_out / resampled_data.size
        print(f"\nVoxels in resampled output outside the raw source's [{lo:.1f}, {hi:.1f}] range: "
              f"{n_out} / {resampled_data.size} ({frac_out:.4%})")
        print(
            "A tiny fraction (<<1%) confined to the extremes points at localized interpolation "
            "overshoot at sharp edges (e.g. the reconstruction-circle boundary) -- not enough on "
            "its own to saturate a whole-body display window. A large fraction, or percentiles "
            "(p1/p5/p95/p99 above) meaningfully shifted relative to the raw source's own "
            "percentiles at the same points, means the perturbation is broad enough that normal "
            "soft tissue is being pushed outside a standard HU display window -- pointing at "
            "resample_volume()'s order=3 zoom (no anti-aliasing pre-filter for downsampling) as "
            "the real root cause."
        )


if __name__ == "__main__":
    main()
