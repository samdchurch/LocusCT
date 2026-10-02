#!/usr/bin/env python3
"""
Resample LocusBench-ED / LocusBench-Onc images and masks onto the same grid
Grounder and finetuned VoxTell train on (1.5 x 1.5 x 3.0mm / 352x352x180 by
default), reusing resample_and_crop.py's image pipeline and
resample_masks.py's mask pipeline unchanged.

Only Grounder's own model and the finetuned-VoxTell checkpoint consume this
offline-resampled grid -- SAT/SegVol/BiomedParse/pretrained VoxTell each do
their own on-the-fly preprocessing and should keep reading LocusBench's raw
images/masks directly.

LocusBench-ED's masks are laid out masks/<FINDING>/<accession>/*.nii.gz
(depth=2, same convention as inhouse_abdominal_ct/ED_TEST_SET/); LocusBench-Onc's
are masks/<accession>/*.nii.gz (depth=1, same convention as
inhouse_abdominal_ct/labels/) -- resample_masks.py's _collect_from_root already
handles both.

Output mirrors the images/ and masks/ subdirectories under one new root per
split, so a single directory still works as both --image-dir and --mask-dir
downstream (matching how the manifest's "image"/"mask" fields are already
prefixed with "images/"/"masks/"):

  <output-root>/LocusBench-ED_resampled/{images,masks}/...
  <output-root>/LocusBench-Onc_resampled/{images,masks}/...

Masks are resampled onto their matching *already-resampled* reference image,
so images must be processed before masks for a given split (--stage images
then --stage masks, or --stage all to run both in order).

Usage
-----
    python resample_locusbench.py --dry-run                    # both splits
    python resample_locusbench.py --split ED --workers 8
    python resample_locusbench.py --split Onc --stage masks --force
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resample_and_crop
import resample_masks

DEFAULT_LOCUSBENCH_ROOT = Path("/path/to/data/LocusBench")
MASK_DEPTH = {"ED": 2, "Onc": 1}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def collect_image_jobs(locusbench_root: Path, split: str, output_root: Path) -> list[tuple[Path, Path]]:
    images_root = locusbench_root / f"LocusBench-{split}" / "images"
    dst_root = output_root / f"LocusBench-{split}_resampled" / "images"
    jobs = []
    for acc_dir in sorted(images_root.iterdir()):
        if not acc_dir.is_dir():
            continue
        for src in sorted(acc_dir.glob("*.nii.gz")):
            jobs.append((src, dst_root / src.relative_to(images_root)))
    return jobs


def collect_mask_jobs(locusbench_root: Path, split: str, output_root: Path) -> list[tuple[Path, Path]]:
    masks_root = locusbench_root / f"LocusBench-{split}" / "masks"
    dst_root = output_root / f"LocusBench-{split}_resampled" / "masks"
    return resample_masks._collect_from_root(masks_root, dst_root, depth=MASK_DEPTH[split], accession=None)


def run_stage(jobs, run_one, desc: str, workers: int, dry_run: bool):
    ok = skip = fail = 0
    if workers == 1 or dry_run:
        for src, dst in tqdm(jobs, desc=desc):
            msg = run_one(src, dst)
            log.info(msg)
            if msg.startswith("SKIP"):
                skip += 1
            elif msg.startswith(("OK", "DRY")):
                ok += 1
            else:
                fail += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, src, dst): (src, dst) for src, dst in jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                src, dst = futures[fut]
                try:
                    msg = fut.result()
                except Exception:
                    log.error("FAIL  %s", src, exc_info=True)
                    fail += 1
                    continue
                log.info(msg)
                if msg.startswith("SKIP"):
                    skip += 1
                elif msg.startswith(("OK", "DRY")):
                    ok += 1
                else:
                    fail += 1
    log.info("%s done.  OK=%d  SKIPPED=%d  FAILED=%d", desc, ok, skip, fail)
    return fail


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["ED", "Onc", "both"], default="both")
    parser.add_argument("--stage", choices=["images", "masks", "all"], default="all")
    parser.add_argument("--locusbench-root", type=Path, default=DEFAULT_LOCUSBENCH_ROOT)
    parser.add_argument("--output-root", type=Path, default=None,
                         help="Default: --locusbench-root's parent directory")
    parser.add_argument("--target-spacing", type=float, nargs=3,
                         default=list(resample_and_crop.DEFAULT_TARGET_SPACING), metavar=("X", "Y", "Z"))
    parser.add_argument("--target-shape", type=int, nargs=3,
                         default=list(resample_and_crop.DEFAULT_TARGET_SHAPE), metavar=("D", "H", "W"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or args.locusbench_root.parent
    target_spacing = np.array(args.target_spacing)
    target_shape = tuple(args.target_shape)
    splits = ["ED", "Onc"] if args.split == "both" else [args.split]

    total_fail = 0
    for split in splits:
        if args.stage in ("images", "all"):
            jobs = collect_image_jobs(args.locusbench_root, split, output_root)
            log.info("[%s] %d image file(s) to resample", split, len(jobs))
            run_one = partial(resample_and_crop.process_file, target_spacing=target_spacing,
                               target_shape=target_shape, dry_run=args.dry_run, force=args.force)
            total_fail += run_stage(jobs, run_one, f"{split} images", args.workers, args.dry_run)

        if args.stage in ("masks", "all"):
            images_root = output_root / f"LocusBench-{split}_resampled" / "images"
            jobs = collect_mask_jobs(args.locusbench_root, split, output_root)
            log.info("[%s] %d mask file(s) to resample", split, len(jobs))
            run_one = partial(resample_masks.process_file, images_root=images_root,
                               dry_run=args.dry_run, force=args.force)
            total_fail += run_stage(jobs, run_one, f"{split} masks", args.workers, args.dry_run)

    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
