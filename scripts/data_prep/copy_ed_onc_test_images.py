#!/usr/bin/env python3
"""
Copies native CT images referenced by the ED and oncology official test set
manifests (official_splits/ed_official_test_data.json,
official_splits/onc_official_test_data.json) into a destination folder,
preserving each manifest entry's own "<AccessionFolder>/<image>.nii.gz"
relative path -- the manifest's "image" field is already in exactly that
format, so no path restructuring is needed, just copying.

Already-copied files are skipped by default (safe to resume/rerun) -- pass
--force to re-copy and overwrite.

Usage
-----
    python copy_ed_onc_test_images.py --output-dir ed_onc_test_images
    python copy_ed_onc_test_images.py --output-dir ed_onc_test_images --dry-run
"""

import argparse
import json
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_MANIFESTS = [
    Path(__file__).resolve().parents[2] / "official_splits" / "ed_official_test_data.json",
    Path(__file__).resolve().parents[2] / "official_splits" / "onc_official_test_data.json",
]
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def collect_image_paths(manifests: list[Path]) -> set[str]:
    images = set()
    for mp in manifests:
        with open(mp) as f:
            data = json.load(f)
        images.update(s["image"] for s in data)
    return images


def copy_one(src: Path, dst: Path, force: bool) -> str:
    if dst.exists() and not force:
        return f"SKIP  (already exists)  {dst}"
    if not src.exists():
        return f"MISSING  {src}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return f"OK    {src} -> {dst}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", type=Path, default=DEFAULT_MANIFESTS,
                         help=f"Manifests to collect image paths from (default: {[str(p) for p in DEFAULT_MANIFESTS]})")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                         help=f"Native image source dir (default: {DEFAULT_IMAGE_DIR})")
    parser.add_argument("--output-dir", required=True,
                         help="Destination folder -- images are copied to <output-dir>/<AccessionFolder>/<image>.nii.gz")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-copy and overwrite files that already exist")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    image_paths = sorted(collect_image_paths(args.manifests))
    logger.info(f"Found {len(image_paths)} unique image(s) across {len(args.manifests)} manifest(s)")

    if args.dry_run:
        for rel in image_paths:
            logger.info(f"DRY   {image_dir / rel}  ->  {output_dir / rel}")
        return

    ok = skip = missing = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(copy_one, image_dir / rel, output_dir / rel, args.force): rel
            for rel in image_paths
        }
        for fut in as_completed(futures):
            msg = fut.result()
            level = logging.WARNING if msg.startswith("MISSING") else logging.INFO
            logger.log(level, msg)
            if msg.startswith("OK"):
                ok += 1
            elif msg.startswith("SKIP"):
                skip += 1
            elif msg.startswith("MISSING"):
                missing += 1

    logger.info("-" * 60)
    logger.info(f"Done. OK={ok}  SKIPPED={skip}  MISSING={missing}")


if __name__ == "__main__":
    main()
