#!/usr/bin/env python3
"""
Isolate which step of resample_and_crop.py's pipeline (zoom vs crop/pad)
causes the intensity shift seen in resampled data (e.g. median shifting from
-970 HU in a raw source file to -748.6 HU in the on-disk resampled file, a
~221 HU shift toward positive).

First pass (order=0/1/3 zoom comparison) ruled out interpolation order: p50
barely moved (-970.0 raw vs -970.4 for every order, 0/1/3 alike). Order=3
only widened the extreme min/max (edge ringing at the reconstruction-circle
boundary), not the bulk of the distribution. So the shift must come from
crop_pad_to_shape() -- which crops/pads to TARGET_SHAPE centered on
centre_of_mass_nonzero(). That function centroids on `data != 0`, but real
HU data is almost never exactly 0.0 (air is -1000, not 0) -- so essentially
every voxel counts as "non-zero", including all the background/padding,
which may not center the crop on the patient the way the name suggests.

This zooms (order=3, matching the pipeline), computes the centroid, prints
where it lands relative to the volume, crops/pads to TARGET_SHAPE, and
reports percentiles at each stage -- plus, if given, the actual on-disk
resampled file's percentiles, to confirm this reproduces the same numbers
before trusting the diagnosis.

Usage
-----
    python check_resample_order.py --accession CASE0000000 --series 2 \
        --raw-root /path/to/data/inhouse_abdominal_ct/nifti \
        --resampled-root /path/to/data/inhouse_abdominal_ct/nifti_resampled
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))  # scripts/data_prep

from resample_and_crop import (
    DEFAULT_TARGET_SHAPE as TARGET_SHAPE,
    DEFAULT_TARGET_SPACING as TARGET_SPACING,
    centre_of_mass_nonzero,
    crop_pad_to_shape,
    get_voxel_spacing,
    resample_volume,
)


def report(label: str, data: np.ndarray) -> None:
    pct = np.percentile(data, [1, 5, 50, 95, 99])
    print(f"  {label:<28} shape={str(data.shape):<18} min={data.min():8.1f}  max={data.max():8.1f}  "
          f"p1={pct[0]:7.1f}  p5={pct[1]:7.1f}  p50={pct[2]:7.1f}  p95={pct[3]:7.1f}  p99={pct[4]:7.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--series", type=int, required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--resampled-root", default=None,
                         help="If given, also load the actual on-disk resampled file for comparison")
    args = parser.parse_args()

    acc_dir = Path(args.raw_root) / args.accession
    candidates = [f for f in acc_dir.glob("*.nii.gz")
                  if f.name.split("_")[0].isdigit() and int(f.name.split("_")[0]) == args.series]
    if not candidates:
        print(f"No series {args.series} file found under {acc_dir}")
        return
    src = candidates[0]
    print(f"Source: {src}\n")

    img = nib.load(str(src))
    data = img.get_fdata(dtype=np.float32)
    orig_spacing = get_voxel_spacing(img)
    print(f"raw shape={data.shape}  spacing={np.round(orig_spacing, 3)}  -> target spacing={TARGET_SPACING}, "
          f"target shape={TARGET_SHAPE}\n")

    report("raw source (no resample)", data)

    resampled, _ = resample_volume(data, orig_spacing, TARGET_SPACING, order=3)
    report("after zoom (order=3)", resampled)

    com = centre_of_mass_nonzero(resampled)
    frac_of_shape = [c / s for c, s in zip(com, resampled.shape)]
    print(f"\ncentre_of_mass_nonzero: {np.round(com, 1)}  "
          f"(volume shape {resampled.shape}, i.e. at {[f'{f:.1%}' for f in frac_of_shape]} along each axis; "
          f"50% would be dead-center)")
    n_nonzero = int(np.sum(resampled != 0))
    print(f"voxels counted as \"non-zero\" for centroid purposes: {n_nonzero} / {resampled.size} "
          f"({n_nonzero / resampled.size:.2%}) -- if this is ~100%, the centroid is really just the "
          f"geometric center, not a true body-center")

    cropped = crop_pad_to_shape(resampled, TARGET_SHAPE, com)
    report("after crop/pad to TARGET_SHAPE", cropped)

    if args.resampled_root:
        resampled_dir = Path(args.resampled_root) / args.accession
        on_disk_candidates = [f for f in resampled_dir.glob("*.nii.gz")
                               if f.name.split("_")[0].isdigit() and int(f.name.split("_")[0]) == args.series]
        if on_disk_candidates:
            on_disk = nib.load(str(on_disk_candidates[0])).get_fdata(dtype=np.float32)
            report("actual on-disk resampled file", on_disk)
        else:
            print(f"\n(no on-disk resampled file found under {resampled_dir} to cross-check against)")

    print(
        "\nIf p50 barely moved after zoom but shifted substantially after crop/pad, that "
        "confirms crop_pad_to_shape()'s centre_of_mass_nonzero()-based centering as the "
        "cause -- likely because it treats nearly all voxels (background included, since "
        "air is -1000 HU, not 0.0) as \"non-zero\", so it isn't really centering on the "
        "patient. The fix would be computing centre_of_mass on a proper body/tissue mask "
        "(e.g. HU > some threshold) instead of data != 0."
    )


if __name__ == "__main__":
    main()
