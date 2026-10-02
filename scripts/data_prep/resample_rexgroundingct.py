#!/usr/bin/env python3
"""
Resample ReXGroundingCT images and their paired segmentation masks directly
from the production dataset mount, using the same recipe as
resample_and_crop.py (images) and resample_masks.py (masks): zoom to
(1.5 x 1.5 x 3.0 mm) spacing, then crop/pad to (352 x 352 x 180) voxels
centred on the centre of mass of non-zero voxels. TARGET_SPACING /
TARGET_SHAPE and the core resample/crop/affine functions are imported
directly from resample_and_crop.py so all pipelines stay in sync.

Unlike resample_masks.py's resample_from_to approach, masks here are
resampled manually per-finding using the *image's* spacing and centre of
mass (not the mask's own affine), because ReXGroundingCT segmentation
files are 4D (finding, H, W, D) and their embedded affine/header is not
trustworthy for spatial resampling (see resample_rexgroundingct_val.py,
which resamples the already-copied local val split the same way). This
script instead walks the production paths directly, matching each image
to its mask by filename stem, and infers the finding count from the
mask array's own shape rather than a manifest. Each finding is split out
and saved as its own single-label 3D file, with the finding's index (its
position along the mask's original first axis) appended to the filename.

Input  : /path/to/data/public_datasets/ReXGroundingCT/images/train_fixed/**/*.nii.gz
         /path/to/data/public_datasets/ReXGroundingCT/images/valid_fixed/**/*.nii.gz
         /path/to/data/public_datasets/ReXGroundingCT/segmentations/<stem>.nii.gz
Output : <output>/images/<split>_fixed/... (mirrors input image layout)
         <output>/segmentations/<stem>_<label>.nii.gz (one file per finding)

A missing mask (e.g. the MICCAI challenge's held-out test split, whose masks are
withheld) does not block the image from being resampled -- only mask splitting is
skipped for that volume, logged as OK_NO_MASK.

Usage
-----
    python resample_rexgroundingct.py
    python resample_rexgroundingct.py --output /path/to/data/rexground_resampled
    python resample_rexgroundingct.py --workers 8
    python resample_rexgroundingct.py --dry-run
"""

import argparse
import json
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from resample_and_crop import (
    AIR_HU,
    TARGET_SHAPE,
    TARGET_SPACING,
    build_affine,
    centre_of_mass_nonzero,
    crop_pad_to_shape,
    get_voxel_spacing,
    resample_volume,
)

IMAGES_ROOT = Path("/path/to/data/public_datasets/ReXGroundingCT/images")
SEGMENTATIONS_ROOT = Path("/path/to/data/public_datasets/ReXGroundingCT/segmentations")
IMAGE_SPLITS = ["train_fixed", "valid_fixed"]
OUTPUT_ROOT = Path("/path/to/data/public_datasets/ReXGroundingCT/resampled")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name


def process_pair(img_src: Path, img_rel: Path, segmentations_root: Path,
                  output_root: Path, dry_run: bool = False) -> str:
    stem = strip_nii_ext(img_src.name)
    mask_src = segmentations_root / f"{stem}.nii.gz"
    has_mask = mask_src.exists()

    out_img_path = output_root / "images" / img_rel
    seg_out_dir = output_root / "segmentations"

    img_done = out_img_path.exists()
    masks_done = any(seg_out_dir.glob(f"{stem}_*.nii.gz")) if has_mask else True
    if img_done and masks_done:
        return f"SKIP  (already exists)  {stem}"

    if dry_run:
        if has_mask:
            return f"DRY   {img_src}  (mask {mask_src.name})  ->  {out_img_path}, {seg_out_dir}/{stem}_<label>.nii.gz"
        return f"DRY   {img_src}  (no mask)  ->  {out_img_path}"

    try:
        img = nib.load(str(img_src))
        img_data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        return f"CORRUPT_IMAGE  {stem}  ({e})"

    if has_mask:
        try:
            mask_img = nib.load(str(mask_src))
            mask_data = np.asanyarray(mask_img.dataobj)
            orig_mask_dtype = mask_img.get_data_dtype()
        except Exception as e:
            return f"CORRUPT_MASK  {stem}  ({e})"

        if mask_data.ndim == 3:
            mask_data = mask_data[np.newaxis, ...]
        num_findings = mask_data.shape[0]

        if mask_data.ndim != 4 or mask_data.shape[1:] != img_data.shape:
            return (f"BAD_MASK_SHAPE  {stem}  mask={mask_data.shape}  "
                    f"expected=(F, {img_data.shape[0]}, {img_data.shape[1]}, {img_data.shape[2]})")

    orig_spacing = get_voxel_spacing(img)

    # ── image: identical recipe to resample_and_crop.py ───────────────────────
    resampled_img, _ = resample_volume(img_data, orig_spacing, TARGET_SPACING, order=3)
    com = centre_of_mass_nonzero(resampled_img)
    final_img = crop_pad_to_shape(resampled_img, TARGET_SHAPE, com, fill_value=AIR_HU)
    new_affine = build_affine(
        original_affine=img.affine,
        original_spacing=orig_spacing,
        target_spacing=TARGET_SPACING,
        resampled_shape=np.array(resampled_img.shape),
        final_shape=TARGET_SHAPE,
        com_resampled=com,
    )

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(final_img, new_affine, header=img.header)
    out_img.header.set_zooms(TARGET_SPACING)
    out_img.header.set_data_shape(TARGET_SHAPE)
    nib.save(out_img, str(out_img_path))

    if not has_mask:
        return (f"OK_NO_MASK  {stem}  img {list(img_data.shape)}@{np.round(orig_spacing, 2)}  ->  "
                f"{list(final_img.shape)}@{TARGET_SPACING}  (no mask)")

    # ── mask: same zoom/crop recipe per finding, nearest-neighbor, same `com` ─
    # each finding is saved as its own 3D file, labeled by its index in the filename
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    for f in range(num_findings):
        resampled_f, _ = resample_volume(mask_data[f].astype(np.float32), orig_spacing, TARGET_SPACING, order=0)
        cropped_f = crop_pad_to_shape(resampled_f, TARGET_SHAPE, com)
        final_f = np.rint(cropped_f).astype(orig_mask_dtype)

        out_mask_path = seg_out_dir / f"{stem}_{f}.nii.gz"
        out_mask = nib.Nifti1Image(final_f, new_affine)
        out_mask.header.set_data_dtype(orig_mask_dtype)
        nib.save(out_mask, str(out_mask_path))

    return (f"OK    {stem}  img {list(img_data.shape)}@{np.round(orig_spacing, 2)}  ->  "
            f"{list(final_img.shape)}@{TARGET_SPACING}  mask F={num_findings} dtype={orig_mask_dtype}")


def collect_jobs(images_root: Path, splits: list[str]) -> list[tuple[Path, Path]]:
    """Collect (src, rel_path) pairs by walking each split directory for *.nii.gz files."""
    jobs = []
    for split in splits:
        split_root = images_root / split
        if not split_root.is_dir():
            log.warning("Split directory not found: %s", split_root)
            continue
        nii_files = sorted(split_root.rglob("*.nii.gz"))
        if not nii_files:
            log.warning("No .nii.gz files found under %s", split_root)
            continue
        for src in nii_files:
            rel = src.relative_to(images_root)
            jobs.append((src, rel))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-root", type=Path, default=IMAGES_ROOT)
    parser.add_argument("--segmentations-root", type=Path, default=SEGMENTATIONS_ROOT)
    parser.add_argument("--splits", nargs="+", default=IMAGE_SPLITS,
                         help="Subdirectories of --images-root to walk (default: train_fixed valid_fixed)")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing files")
    args = parser.parse_args()

    jobs = collect_jobs(args.images_root, args.splits)
    log.info("Found %d image file(s) across %d split(s)", len(jobs), len(args.splits))

    if not jobs:
        log.error("No files found under %s", args.images_root)
        sys.exit(1)

    ok = skip = no_mask = corrupt = bad_shape = fail = 0
    failures: list[dict] = []

    def _record(stem: str, msg: str) -> None:
        nonlocal ok, skip, no_mask, corrupt, bad_shape, fail
        if msg.startswith("SKIP"):
            skip += 1
        elif msg.startswith("OK_NO_MASK"):
            no_mask += 1
        elif msg.startswith("CORRUPT"):
            corrupt += 1
            failures.append({"name": stem, "status": "corrupt", "reason": msg})
        elif msg.startswith("BAD_MASK_SHAPE"):
            bad_shape += 1
            failures.append({"name": stem, "status": "bad_mask_shape", "reason": msg})
        elif msg.startswith("DRY"):
            pass
        else:
            ok += 1

    if args.workers == 1 or args.dry_run:
        for src, rel in jobs:
            stem = strip_nii_ext(src.name)
            try:
                msg = process_pair(src, rel, args.segmentations_root, args.output, dry_run=args.dry_run)
                level = logging.WARNING if not (msg.startswith("OK") or msg.startswith("SKIP") or msg.startswith("DRY")) else logging.INFO
                log.log(level, msg)
                _record(stem, msg)
            except Exception:
                log.error("FAIL  %s\n%s", stem, traceback.format_exc())
                fail += 1
                failures.append({"name": stem, "status": "failed", "reason": traceback.format_exc()})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_pair, src, rel, args.segmentations_root, args.output, args.dry_run): src
                       for src, rel in jobs}
            for fut in as_completed(futures):
                src = futures[fut]
                stem = strip_nii_ext(src.name)
                try:
                    msg = fut.result()
                    level = logging.WARNING if not (msg.startswith("OK") or msg.startswith("SKIP") or msg.startswith("DRY")) else logging.INFO
                    log.log(level, msg)
                    _record(stem, msg)
                except Exception:
                    log.error("FAIL  %s\n%s", stem, traceback.format_exc())
                    fail += 1
                    failures.append({"name": stem, "status": "failed", "reason": traceback.format_exc()})

    log.info("-" * 60)
    log.info("Done.  OK=%d  SKIPPED=%d  OK_NO_MASK=%d  CORRUPT=%d  BAD_MASK_SHAPE=%d  FAILED=%d",
              ok, skip, no_mask, corrupt, bad_shape, fail)

    if failures and not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        failures_path = args.output / "failures.json"
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
        log.info("Wrote %d failed record(s) to %s", len(failures), failures_path)

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
