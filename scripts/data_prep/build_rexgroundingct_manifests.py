#!/usr/bin/env python3
"""
Build official_splits manifests for the ReXGroundingCT MICCAI challenge dataset,
one record per (volume, finding) pair, matching the GrounderDataset manifest format
used by the rest of official_splits (image, mask, sentence, region, finding).

The challenge manifest's three top-level keys (train/val/test) map 1:1 to the three
output files. Image paths are resolved the same way copy_rexgroundingct_val.py
resolves them: CT-RATE's own split/patient/scan naming is embedded in each volume's
"name" (e.g. "train_13082_a_1.nii.gz" lives under images/train_fixed/train_13082/
train_13082_a_1/), regardless of which challenge split it's assigned to. Mask paths
follow resample_rexgroundingct.py's per-finding naming: segmentations/<stem>_<finding_idx>.nii.gz.

The test split ships with no segmentation masks (withheld by the challenge for the
hidden leaderboard), so test records get "mask": null.

Image/mask paths are written relative to PATH_STRIP_ROOT (default: the
/path/to/data/public_datasets mount shared by all public dataset mirrors), matching
the relative-path convention of the other official_splits manifests. Point
data.image_dir/data.mask_dir at PATH_STRIP_ROOT to load these entries.

Input  : /path/to/data/public_datasets/ReXGroundingCT/MICCAI_challenge_dataset.json
Images : /path/to/data/public_datasets/ReXGroundingCT/resampled/images/<split>_fixed/...
Masks  : /path/to/data/public_datasets/ReXGroundingCT/resampled/segmentations/<stem>_<idx>.nii.gz
Output : official_splits/ReXGroundingCT_train.json
         official_splits/ReXGroundingCT_val.json
         official_splits/ReXGroundingCT_test.json

Usage
-----
    python build_rexgroundingct_manifests.py
    python build_rexgroundingct_manifests.py --dry-run
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

MANIFEST_PATH = Path("/path/to/data/public_datasets/ReXGroundingCT/MICCAI_challenge_dataset.json")
RESAMPLED_ROOT = Path("/path/to/data/public_datasets/ReXGroundingCT/resampled")
IMAGES_ROOT = RESAMPLED_ROOT / "images"
SEGMENTATIONS_ROOT = RESAMPLED_ROOT / "segmentations"
OFFICIAL_SPLITS_DIR = Path(__file__).resolve().parents[2] / "official_splits"
PATH_STRIP_ROOT = Path("/path/to/data/public_datasets")

REGION = "Chest"

# CT-RATE volume naming: <split>_<patient>_<scan>_<recon>, e.g. "train_13082_a_1"
STEM_RE = re.compile(r"^(?P<split>[A-Za-z]+)_(?P<patient>\d+)_(?P<scan>[A-Za-z0-9]+)_(?P<recon>\d+)$")

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


def resolve_image_path(stem: str, images_root: Path) -> Path | None:
    """Construct a volume's resampled image path from its stem (CT-RATE naming)."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    patient_dir = f"{m['split']}_{m['patient']}"
    scan_dir = f"{patient_dir}_{m['scan']}"
    return images_root / f"{m['split']}_fixed" / patient_dir / scan_dir / f"{stem}.nii.gz"


def relativize(path: Path, strip_root: Path) -> str:
    try:
        return path.relative_to(strip_root).as_posix()
    except ValueError:
        log.warning("%s is not relative to %s; writing an absolute path instead", path, strip_root)
        return path.as_posix()


def build_split_records(records: list[dict], split_name: str, images_root: Path,
                         segmentations_root: Path, strip_root: Path) -> tuple[list[dict], list[dict]]:
    out_records = []
    missing = []

    for rec in records:
        name = rec.get("name")
        findings = rec.get("findings", {})
        if not name or not findings:
            missing.append({"reason": "no_name_or_findings", "record": rec})
            continue

        stem = strip_nii_ext(name)
        img_path = resolve_image_path(stem, images_root)
        if img_path is None:
            missing.append({"reason": "unparseable_name", "name": name})
            continue
        if not img_path.exists():
            missing.append({"reason": "image_not_found", "name": name, "expected_path": str(img_path)})
            continue

        for idx_str, sentence in findings.items():
            if split_name == "test":
                mask_field = None
            else:
                mask_path = segmentations_root / f"{stem}_{idx_str}.nii.gz"
                if not mask_path.exists():
                    missing.append({"reason": "mask_not_found", "name": name, "finding_idx": idx_str,
                                     "expected_path": str(mask_path)})
                    continue
                mask_field = relativize(mask_path, strip_root)

            out_records.append({
                "image": relativize(img_path, strip_root),
                "mask": mask_field,
                "sentence": sentence,
                "region": REGION,
                "finding": None,
            })

    return out_records, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--images-root", type=Path, default=IMAGES_ROOT)
    parser.add_argument("--segmentations-root", type=Path, default=SEGMENTATIONS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OFFICIAL_SPLITS_DIR)
    parser.add_argument("--strip-root", type=Path, default=PATH_STRIP_ROOT,
                         help="Written image/mask paths are relativized against this root")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing output files")
    args = parser.parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        sys.exit(1)

    with open(args.manifest) as f:
        manifest = json.load(f)

    all_missing = []
    for split_name in ("train", "val", "test"):
        records = manifest.get(split_name)
        if records is None:
            log.warning("No %r key in manifest, skipping", split_name)
            continue

        out_records, missing = build_split_records(records, split_name, args.images_root, args.segmentations_root,
                                                     args.strip_root)
        all_missing.extend({"split": split_name, **m} for m in missing)

        log.info("[%s] %d source record(s) -> %d manifest entries, %d skipped (missing)",
                  split_name, len(records), len(out_records), len(missing))

        if not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / f"ReXGroundingCT_{split_name}.json"
            with open(out_path, "w") as f:
                json.dump(out_records, f, indent=4)
            log.info("Wrote %d record(s) to %s", len(out_records), out_path)

    if all_missing:
        log.warning("%d total skipped record(s) across all splits", len(all_missing))
        if not args.dry_run:
            missing_path = args.output_dir / "rexgroundingct_manifest_missing.json"
            with open(missing_path, "w") as f:
                json.dump(all_missing, f, indent=2)
            log.info("Wrote skipped-record details to %s", missing_path)


if __name__ == "__main__":
    main()
