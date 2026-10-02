#!/usr/bin/env python3
"""
Build a small benchmark subset to compare .nii.gz vs uncompressed .nii
training throughput, without touching the full dataset.

Selects the first N samples (in manifest order, skipping any with missing
files) from a manifest, converts their image/mask NIfTI files to
uncompressed .nii under a new directory (de-duplicating files shared by
multiple samples), and writes two manifests over the *same* N samples:

    <out_manifest_prefix>_gz.json   -> original .nii.gz files, unchanged
    <out_manifest_prefix>_nii.json  -> converted .nii files, new directory

Train on each manifest and compare epoch time / it-per-sec to see whether
converting the full dataset to .nii is worth the disk space.

Usage
-----
    python make_nii_benchmark.py
    python make_nii_benchmark.py --n 1000 --workers 8
    python make_nii_benchmark.py --dry-run
"""

import argparse
import json
import logging
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEFAULT_MANIFEST = "official_splits/all_data_train.json"
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti_resampled"
DEFAULT_MASK_DIR = "/path/to/data/inhouse_abdominal_ct/labels_resampled"
DEFAULT_OUT_DIR = "/path/to/data/inhouse_abdominal_ct/nii_benchmark"
DEFAULT_OUT_MANIFEST_PREFIX = "official_splits/benchmark_1000"


def to_nii_name(rel_path: str) -> str:
    """'foo/bar.nii.gz' -> 'foo/bar.nii'; leaves already-uncompressed paths alone."""
    return rel_path[:-3] if rel_path.endswith(".gz") else rel_path


def convert_one(src: Path, dst: Path, dry_run: bool = False) -> str:
    """Convert a single NIfTI file to uncompressed .nii at dst (no resampling)."""
    if dst.exists():
        return f"SKIP  (already exists)  {dst}"
    if dry_run:
        return f"DRY   {src} -> {dst}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(src))
    nib.save(img, str(dst))
    return f"OK    {src.name} -> {dst}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                         help="Root for converted .nii files (images/ and masks/ subfolders created under it)")
    parser.add_argument("--out_manifest_prefix", default=DEFAULT_OUT_MANIFEST_PREFIX,
                         help="Writes <prefix>_gz.json and <prefix>_nii.json")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)

    with open(args.manifest) as f:
        all_samples = json.load(f)

    selected = []
    for sample in all_samples:
        if (image_dir / sample["image"]).exists() and (mask_dir / sample["mask"]).exists():
            selected.append(sample)
            if len(selected) == args.n:
                break

    if len(selected) < args.n:
        log.warning("Only found %d/%d samples with files present", len(selected), args.n)
    log.info("Selected %d samples", len(selected))

    # Map each source file to its converted destination, de-duplicating
    # images/masks that are shared across multiple samples (e.g. one CT
    # referenced by several findings/expressions).
    jobs: dict[Path, Path] = {}
    nii_samples = []
    for sample in selected:
        img_rel_nii = to_nii_name(sample["image"])
        msk_rel_nii = to_nii_name(sample["mask"])
        jobs[image_dir / sample["image"]] = out_dir / "images" / img_rel_nii
        jobs[mask_dir / sample["mask"]] = out_dir / "masks" / msk_rel_nii

        nii_sample = dict(sample)
        nii_sample["image"] = img_rel_nii
        nii_sample["mask"] = msk_rel_nii
        nii_samples.append(nii_sample)

    log.info("Converting %d unique file(s) (de-duplicated from %d image/mask refs)",
              len(jobs), len(selected) * 2)

    ok = skip = fail = 0
    items = list(jobs.items())
    if args.workers == 1 or args.dry_run:
        for src, dst in items:
            try:
                msg = convert_one(src, dst, dry_run=args.dry_run)
                log.info(msg)
                if msg.startswith("SKIP"):
                    skip += 1
                else:
                    ok += 1
            except Exception:
                log.error("FAIL  %s\n%s", src, traceback.format_exc())
                fail += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(convert_one, src, dst): src for src, dst in items}
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    msg = fut.result()
                    log.info(msg)
                    if msg.startswith("SKIP"):
                        skip += 1
                    else:
                        ok += 1
                except Exception:
                    log.error("FAIL  %s\n%s", src, traceback.format_exc())
                    fail += 1

    log.info("-" * 60)
    log.info("Conversion done.  OK=%d  SKIPPED=%d  FAILED=%d", ok, skip, fail)

    if not args.dry_run:
        gz_manifest_path = Path(f"{args.out_manifest_prefix}_gz.json")
        nii_manifest_path = Path(f"{args.out_manifest_prefix}_nii.json")
        gz_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gz_manifest_path, "w") as f:
            json.dump(selected, f, indent=2)
        with open(nii_manifest_path, "w") as f:
            json.dump(nii_samples, f, indent=2)
        log.info("Wrote %s and %s", gz_manifest_path, nii_manifest_path)

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
