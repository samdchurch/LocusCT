#!/usr/bin/env python3
"""
Resample the ReXGroundingCT validation images and masks the same way
training data is resampled by resample_and_crop.py: zoom to
(1.5 x 1.5 x 3.0 mm) spacing, then crop/pad to (352 x 352 x 180) voxels
centred on the centre of mass of non-zero voxels. TARGET_SPACING /
TARGET_SHAPE and the core resample/crop/affine functions are imported
directly from resample_and_crop.py so both pipelines can't drift apart.

Images use cubic interpolation (order=3), same as resample_and_crop.py.
Masks use nearest-neighbor (order=0) to preserve discrete labels, and
are cropped using the SAME centre-of-mass computed from the resampled
image (not recomputed from the mask) so image and mask stay aligned.

Segmentation masks are 4D volumes shaped (F, H, W, D) — findings on the
array's first axis, confirmed via rexrank_eval.py's own `gt[i]` indexing
over axis 0 — NOT the standard NIfTI spatial-first convention. Because
of this, the mask file's own embedded affine/header isn't trustworthy
for spatial resampling; each finding's 3D slice is instead resampled
using the *paired CT image's* spacing (the physical grid the mask was
built on), the same way resample_masks.py resamples masks onto their
reference image's grid for the other (inhouse_abdominal_ct) dataset.

Unverified: as of writing, the local segmentations mirror
(.../ReXGroundingCT/segmentations) contained unpulled Git LFS pointer
stubs rather than real NIfTI content, so this axis-order assumption
could not be checked against real data. Use --limit 1 after pulling
real LFS content to sanity-check one example (check the OK log line's
mask shape/dtype) before processing the full batch.

Input  : ReXGroundingCT_val/<split>_fixed/... (images, from copy_rexgroundingct_val.py)
         ReXGroundingCT_val/segmentations/<name>.nii.gz (masks)
         ReXGroundingCT_val/val_manifest.json (record list: name, findings, shape, ...)
Output : ReXGroundingCT_val_resampled/images/<name>.nii.gz
         ReXGroundingCT_val_resampled/segmentations/<name>.nii.gz

Usage
-----
    python resample_rexgroundingct_val.py
    python resample_rexgroundingct_val.py --limit 1       # sanity-check one example
    python resample_rexgroundingct_val.py --workers 8
    python resample_rexgroundingct_val.py --dry-run
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

from copy_rexgroundingct_val import resolve_path, strip_nii_ext
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

VAL_ROOT = Path("ReXGroundingCT_val")
OUTPUT_ROOT = Path("ReXGroundingCT_val_resampled")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def process_pair(record: dict, val_root: Path, output_root: Path, dry_run: bool = False) -> str:
    stem = strip_nii_ext(record["name"])
    img_path = resolve_path(stem, val_root)
    mask_path = val_root / "segmentations" / f"{stem}.nii.gz"

    out_img_path = output_root / "images" / f"{stem}.nii.gz"
    out_mask_path = output_root / "segmentations" / f"{stem}.nii.gz"

    if out_img_path.exists() and out_mask_path.exists():
        return f"SKIP  (already exists)  {stem}"

    if img_path is None or not img_path.exists():
        return f"MISSING_IMAGE  {stem}"
    if not mask_path.exists():
        return f"MISSING_MASK  {stem}"

    if dry_run:
        return f"DRY   {stem}  ->  {out_img_path}, {out_mask_path}"

    try:
        img = nib.load(str(img_path))
        img_data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        return f"CORRUPT_IMAGE  {stem}  ({e})"

    try:
        mask_img = nib.load(str(mask_path))
        mask_data = np.asanyarray(mask_img.dataobj)
        orig_mask_dtype = mask_img.get_data_dtype()
    except Exception as e:
        return f"CORRUPT_MASK  {stem}  ({e})"

    num_findings = len(record.get("findings", {}))
    if mask_data.ndim == 3 and num_findings == 1:
        mask_data = mask_data[np.newaxis, ...]

    if mask_data.ndim != 4 or mask_data.shape[0] != num_findings or mask_data.shape[1:] != img_data.shape:
        return (f"BAD_MASK_SHAPE  {stem}  mask={mask_data.shape}  "
                f"expected=({num_findings}, {img_data.shape[0]}, {img_data.shape[1]}, {img_data.shape[2]})")

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

    # ── mask: same zoom/crop recipe per finding, nearest-neighbor, same `com` ─
    final_mask = np.zeros((num_findings, *TARGET_SHAPE), dtype=orig_mask_dtype)
    for f in range(num_findings):
        resampled_f, _ = resample_volume(mask_data[f].astype(np.float32), orig_spacing, TARGET_SPACING, order=0)
        cropped_f = crop_pad_to_shape(resampled_f, TARGET_SHAPE, com)
        final_mask[f] = np.rint(cropped_f).astype(orig_mask_dtype)

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(final_img, new_affine, header=img.header)
    out_img.header.set_zooms(TARGET_SPACING)
    out_img.header.set_data_shape(TARGET_SHAPE)
    nib.save(out_img, str(out_img_path))

    out_mask_path.parent.mkdir(parents=True, exist_ok=True)
    out_mask = nib.Nifti1Image(final_mask, new_affine)
    out_mask.header.set_data_dtype(orig_mask_dtype)
    nib.save(out_mask, str(out_mask_path))

    return (f"OK    {stem}  img {list(img_data.shape)}@{np.round(orig_spacing, 2)}  ->  "
            f"{list(final_img.shape)}@{TARGET_SPACING}  mask F={num_findings} dtype={orig_mask_dtype}")


def load_records(manifest_path: Path, limit: int | None) -> list[dict]:
    with open(manifest_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise TypeError(f"Expected a list of records in {manifest_path}, got {type(records)}")
    return records[:limit] if limit else records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val-root", type=Path, default=VAL_ROOT,
                         help="Folder produced by copy_rexgroundingct_val.py")
    parser.add_argument("--manifest", type=Path, default=None,
                         help="Defaults to <val-root>/val_manifest.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N records (for sanity-checking)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing files")
    args = parser.parse_args()

    manifest_path = args.manifest or (args.val_root / "val_manifest.json")
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    records = load_records(manifest_path, args.limit)
    log.info("Loaded %d record(s) from %s", len(records), manifest_path)

    ok = skip = missing = corrupt = bad_shape = fail = 0
    failures: list[dict] = []

    def _record(record: dict, msg: str) -> None:
        nonlocal ok, skip, missing, corrupt, bad_shape, fail
        if msg.startswith("SKIP"):
            skip += 1
        elif msg.startswith("MISSING"):
            missing += 1
            failures.append({"name": record["name"], "status": "missing", "reason": msg})
        elif msg.startswith("CORRUPT"):
            corrupt += 1
            failures.append({"name": record["name"], "status": "corrupt", "reason": msg})
        elif msg.startswith("BAD_MASK_SHAPE"):
            bad_shape += 1
            failures.append({"name": record["name"], "status": "bad_mask_shape", "reason": msg})
        elif msg.startswith("DRY"):
            pass
        else:
            ok += 1

    if args.workers == 1 or args.dry_run:
        for record in records:
            try:
                msg = process_pair(record, args.val_root, args.output, dry_run=args.dry_run)
                level = logging.WARNING if not (msg.startswith("OK") or msg.startswith("SKIP") or msg.startswith("DRY")) else logging.INFO
                log.log(level, msg)
                _record(record, msg)
            except Exception:
                log.error("FAIL  %s\n%s", record.get("name"), traceback.format_exc())
                fail += 1
                failures.append({"name": record.get("name"), "status": "failed", "reason": traceback.format_exc()})
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_pair, record, args.val_root, args.output, args.dry_run): record
                       for record in records}
            for fut in as_completed(futures):
                record = futures[fut]
                try:
                    msg = fut.result()
                    level = logging.WARNING if not (msg.startswith("OK") or msg.startswith("SKIP") or msg.startswith("DRY")) else logging.INFO
                    log.log(level, msg)
                    _record(record, msg)
                except Exception:
                    log.error("FAIL  %s\n%s", record.get("name"), traceback.format_exc())
                    fail += 1
                    failures.append({"name": record.get("name"), "status": "failed", "reason": traceback.format_exc()})

    log.info("-" * 60)
    log.info("Done.  OK=%d  SKIPPED=%d  MISSING=%d  CORRUPT=%d  BAD_MASK_SHAPE=%d  FAILED=%d",
              ok, skip, missing, corrupt, bad_shape, fail)

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
