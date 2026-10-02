#!/usr/bin/env python3
"""
Resample binary NIfTI segmentation masks onto the same voxel grid (spacing,
shape, affine) as their already-resampled CT images, using nearest-neighbor
interpolation to preserve binary values.

Handles two source layouts, both matched to their corresponding resampled
image by accession folder and leading series number in the filename:

  Main dataset:
    labels/<accession>/mask_<series>_<id>_<id2>.nii.gz
        <-> nifti_resampled/<accession>/<series>__*.nii.gz
        -> labels_resampled/<accession>/*.nii.gz

  ED official test set:
    ED_TEST_SET/<category>/<accession>/Struct_<CATEGORY>_<series>_<slice>_<annoIdx>.nii.gz
        <-> nifti_resampled/<accession>/<series>__*.nii.gz
        -> ED_TEST_SET_resampled/<category>/<accession>/*.nii.gz

Run resample_and_crop.py first so the reference images exist.

--images-root / --output-root / --ed-test-output-root default to the paths
below, so any existing invocation with no new flags is unaffected. Point
--images-root at an alternate image grid (e.g. resample_and_crop.py's
--output-root 192^3 folder) and pass matching --output-root/
--ed-test-output-root folders to resample masks onto that grid instead,
without touching the default labels_resampled/ output.

Input  : /path/to/data/inhouse_abdominal_ct/labels/<accession>/*.nii.gz
         /path/to/data/inhouse_abdominal_ct/ED_TEST_SET/<category>/<accession>/*.nii.gz
Images : /path/to/data/inhouse_abdominal_ct/nifti_resampled/<accession>/<series>__*.nii.gz
Output : /path/to/data/inhouse_abdominal_ct/labels_resampled/<accession>/*.nii.gz
         /path/to/data/inhouse_abdominal_ct/ED_TEST_SET_resampled/<category>/<accession>/*.nii.gz

Usage
-----
    python resample_masks.py              # process everything (both sources)
    python resample_masks.py --workers 8  # parallelise over N cores
    python resample_masks.py --dry-run    # print what would be done

    # Match an alternate 192^3 image grid, into its own label folders:
    python resample_masks.py \\
        --images-root /path/to/data/inhouse_abdominal_ct/nifti_resampled_192 \\
        --output-root /path/to/data/inhouse_abdominal_ct/labels_resampled_192 \\
        --ed-test-output-root /path/to/data/inhouse_abdominal_ct/ED_TEST_SET_resampled_192
"""

import argparse
import json
import logging
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# ── paths ──────────────────────────────────────────────────────────────────────
BASE               = Path("/path/to/data/inhouse_abdominal_ct")
LABELS_ROOT        = BASE / "labels"
ED_TEST_ROOT       = BASE / "ED_TEST_SET"
DEFAULT_IMAGES_ROOT        = BASE / "nifti_resampled"
DEFAULT_OUTPUT_ROOT        = BASE / "labels_resampled"
DEFAULT_ED_TEST_OUTPUT_ROOT = BASE / "ED_TEST_SET_resampled"

# Matches either naming scheme, capturing the leading series number:
#   mask_2_118_0.nii.gz / mask_box_602_88_0.nii.gz  -> "2" / "602"
#   Struct_ABSCESS_2_118_0.nii.gz                   -> "2"
MASK_SERIES_RE = re.compile(r"^(?:mask_(?:box_)?|Struct_[A-Za-z_ ]+_)(\d+)_")

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def process_file(src: Path, dst: Path, images_root: Path, dry_run: bool = False, force: bool = False) -> str:
    """
    Resample a single binary mask onto the grid of its corresponding
    resampled image. Returns a short status string beginning with one of:
      OK / SKIP / DRY / EMPTY / NO_REF_REGEX / NO_REF_IMAGE
    Exceptions propagate to the caller and are counted separately as FAIL.
    """
    if dst.exists() and not force:
        return f"SKIP  (already exists)  {dst.name}"

    m = MASK_SERIES_RE.match(src.name)
    if not m:
        return (f"NO_REF_REGEX  {src.parent.name}/{src.name}  "
                f"(filename does not match mask_<series>_ or Struct_<CATEGORY>_<series>_ pattern)")

    series = m.group(1)
    acc_dir = images_root / src.parent.name
    candidates = sorted(acc_dir.glob(f"{series}__*.nii.gz"))
    if not candidates:
        return (f"NO_REF_IMAGE  {src.parent.name}/{src.name}  "
                f"(no {series}__*.nii.gz found under {acc_dir})")
    ref = candidates[0]

    if dry_run:
        return f"DRY   {src}  (ref {ref.name})  →  {dst}"

    mask_img = nib.load(str(src))
    ref_img = nib.load(str(ref))

    resampled = resample_from_to(mask_img, ref_img, order=0)

    out_data = np.asanyarray(resampled.dataobj).astype(np.uint8)
    out_img = nib.Nifti1Image(out_data, resampled.affine, header=resampled.header)
    out_img.header.set_data_dtype(np.uint8)

    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(dst))

    if out_data.max() == 0:
        return (f"EMPTY  {src.parent.name}/{src.name}  →  {ref.name}  "
                f"{list(out_data.shape)}  (all-zero mask after resampling)")

    return f"OK    {src.parent.name}/{src.name}  →  {ref.name}  {list(out_data.shape)}"


def _collect_from_root(
    root: Path, output_root: Path, depth: int, accession: str | None
) -> list[tuple[Path, Path]]:
    """
    Collect (src, dst) pairs under one label tree.
      depth=1: root/<accession>/*.nii.gz             (main dataset's labels/)
      depth=2: root/<category>/<accession>/*.nii.gz  (ED_TEST_SET/)
    """
    if not root.is_dir():
        log.warning("Root not found, skipping: %s", root)
        return []

    pattern = "/".join(["*"] * depth)
    acc_dirs = [p for p in root.glob(pattern) if p.is_dir()]
    if accession:
        acc_dirs = [p for p in acc_dirs if p.name == accession]

    jobs = []
    for acc_dir in acc_dirs:
        nii_files = sorted(acc_dir.glob("*.nii.gz"))
        if not nii_files:
            log.warning("No .nii.gz files in %s — skipping", acc_dir)
            continue
        for src in nii_files:
            rel = src.relative_to(root)
            dst = output_root / rel
            jobs.append((src, dst))
    return jobs


def collect_jobs(
    output_root: Path, ed_test_output_root: Path, accession: str | None = None,
) -> list[tuple[Path, Path]]:
    """
    Collect (src, dst) pairs across both the main-dataset labels/ tree and the
    ED_TEST_SET/ tree.  If `accession` is given, only process that single
    accession folder (used by Slurm array tasks) -- checked against whichever
    root(s) actually contain it.
    """
    jobs = []
    jobs += _collect_from_root(LABELS_ROOT, output_root, depth=1, accession=accession)
    jobs += _collect_from_root(ED_TEST_ROOT, ed_test_output_root, depth=2, accession=accession)
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers",   type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--force",     action="store_true",
                        help="Overwrite existing output files instead of skipping them "
                             "(e.g. re-resample labels_resampled/ onto a regenerated image grid)")
    parser.add_argument("--accession", type=str, default=None,
                        help="Process only this accession folder name (used by Slurm array jobs)")
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT,
                        help=f"Reference resampled-image directory (default: {DEFAULT_IMAGES_ROOT})")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help=f"Main-dataset label output directory (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--ed-test-output-root", type=Path, default=DEFAULT_ED_TEST_OUTPUT_ROOT,
                        help=f"ED_TEST_SET label output directory (default: {DEFAULT_ED_TEST_OUTPUT_ROOT})")
    args = parser.parse_args()

    images_root = args.images_root
    output_root = args.output_root
    ed_test_output_root = args.ed_test_output_root

    jobs = collect_jobs(output_root, ed_test_output_root, accession=args.accession)
    log.info("Found %d mask file(s) across %d accession folder(s)",
             len(jobs),
             len({j[0].parent for j in jobs}))

    if not jobs:
        log.error("No files found under %s or %s", LABELS_ROOT, ED_TEST_ROOT)
        sys.exit(1)

    counts: dict[str, int] = {"OK": 0, "SKIP": 0, "EMPTY": 0,
                               "NO_REF_REGEX": 0, "NO_REF_IMAGE": 0, "FAIL": 0}
    no_ref_image: list[dict] = []

    def _record(src: Path, msg: str) -> None:
        for key in counts:
            if msg.startswith(key):
                counts[key] += 1
                break
        else:
            counts["FAIL"] += 1

        if msg.startswith("NO_REF_IMAGE"):
            m = MASK_SERIES_RE.match(src.name)
            no_ref_image.append({
                "accession": src.parent.name,
                "series": m.group(1) if m else None,
            })

    # ~20 updates over the whole run, all to stdout alongside the per-file
    # logs -- unlike tqdm (stderr, ends up in a separate SLURM .out/.err
    # file and doesn't render as a live bar in a plain log file anyway).
    progress_every = max(1, len(jobs) // 20)

    def _log_progress(done: int, total: int) -> None:
        if done % progress_every == 0 or done == total:
            log.info("Progress: %d/%d (%.1f%%)", done, total, 100 * done / total)

    if args.workers == 1 or args.dry_run:
        for i, (src, dst) in enumerate(jobs, 1):
            try:
                msg = process_file(src, dst, images_root, dry_run=args.dry_run, force=args.force)
                level = logging.WARNING if msg.startswith(("EMPTY", "NO_REF")) else logging.INFO
                log.log(level, msg)
                _record(src, msg)
            except Exception:
                log.error("FAIL  %s\n%s", src, traceback.format_exc())
                counts["FAIL"] += 1
            _log_progress(i, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, src, dst, images_root, force=args.force): src for src, dst in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                src = futures[fut]
                try:
                    msg = fut.result()
                    level = logging.WARNING if msg.startswith(("EMPTY", "NO_REF")) else logging.INFO
                    log.log(level, msg)
                    _record(src, msg)
                except Exception:
                    log.error("FAIL  %s\n%s", src, traceback.format_exc())
                    counts["FAIL"] += 1
                _log_progress(i, len(jobs))

    log.info("─" * 60)
    log.info("Done.  OK=%d  SKIP=%d  EMPTY=%d  NO_REF_REGEX=%d  NO_REF_IMAGE=%d  FAIL=%d",
             counts["OK"], counts["SKIP"], counts["EMPTY"],
             counts["NO_REF_REGEX"], counts["NO_REF_IMAGE"], counts["FAIL"])

    if no_ref_image and not args.dry_run:
        suffix = f"_{args.accession}" if args.accession else ""
        no_ref_path = BASE / f"no_ref_image{suffix}.json"
        with open(no_ref_path, "w") as f:
            json.dump(no_ref_image, f, indent=2)
        log.info("Wrote %d NO_REF_IMAGE case(s) to %s", len(no_ref_image), no_ref_path)

    if counts["FAIL"] or counts["EMPTY"] or counts["NO_REF_REGEX"] or counts["NO_REF_IMAGE"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
