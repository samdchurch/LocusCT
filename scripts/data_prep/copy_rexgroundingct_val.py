#!/usr/bin/env python3
"""
Copy the ReXGroundingCT train- and validation-split CT volumes out of the
local CT-RATE mirror into a standalone folder for transfer to another server.

The MICCAI challenge manifest links each grounding example to its source
CT-RATE volume via a "name" field (CT-RATE's own naming, e.g.
"train_13082_a_1.nii.gz"). Note the challenge's "val" split draws on
volumes from CT-RATE's *train* pool, not CT-RATE's own valid split —
that's just how CT-RATE happened to originally partition the case, it
has nothing to do with the challenge's own train/val assignment.

To avoid touching anything beyond the specific listed files (no directory
walk over train_fixed's much larger tree), each volume's path is
constructed directly from its name using CT-RATE's fixed layout:
    <dataset_root>/<split>_fixed/<split>_<patient>/<split>_<patient>_<scan>/<name>
and only that exact path is checked/copied — nothing else under
train_fixed or valid_fixed is read. Segmentation masks live in a flat
directory keyed by the same <name>, one file per scan (not per-finding):
    <segmentations_root>/<name>
This script:
  1. Loads the manifest and, for each requested split (default: "train"
     and "val"; pass --splits test to also/instead include "test"), pulls
     out that split's records (top-level key matching the split's aliases,
     or a list with a matching "split" field). The challenge withholds
     segmentation masks for the test split, so masks are never looked up
     for it regardless of --no-segmentations.
  2. Resolves each record's CT volume and segmentation mask to their
     exact paths and copies them into the output folder (images under
     their path relative to dataset_root, masks under "segmentations/").
     Volumes are deduplicated by name across both splits combined, so a
     volume referenced by both is only copied once. A file already present
     in the output at the same size as its source is skipped, so a rerun
     after an interrupted copy only fetches what's missing.
  3. Writes a filtered copy of each split's manifest (e.g. train_manifest.json,
     val_manifest.json) into the output folder so it's self-contained after
     transfer.
  4. Logs any manifest entries whose image or mask couldn't be found to
     unmatched.json in the output folder.

Assumptions — verify against the printed diagnostics on first run:
  - Manifest top level is a dict keyed by split name (train/training,
    val/valid/validation).
  - Each record has a "name" field naming the source .nii.gz, formatted
    as "<split>_<patient>_<scan>_<recon>.nii.gz" (CT-RATE convention).
Override with --split-key-train / --split-key-val / --id-field if those
don't hold, or --splits to copy only one split.

Input   : /path/to/data/public_datasets/ReXGroundingCT/MICCAI_challenge_dataset.json
Images  : /path/to/data/public_datasets/CT-RATE/dataset/<split>_fixed (per record)
Masks   : /path/to/data/public_datasets/ReXGroundingCT/segmentations (per record)
Output  : ./ReXGroundingCT_val (default; override with --output)

Usage
-----
    python copy_rexgroundingct_val.py
    python copy_rexgroundingct_val.py --output /path/to/data/rexground_val
    python copy_rexgroundingct_val.py --splits val
    python copy_rexgroundingct_val.py --splits test
    python copy_rexgroundingct_val.py --dry-run
"""

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

MANIFEST_PATH = Path("/path/to/data/public_datasets/ReXGroundingCT/MICCAI_challenge_dataset.json")
CT_RATE_DATASET_ROOT = Path("/path/to/data/public_datasets/CT-RATE/dataset")
SEGMENTATIONS_ROOT = Path("/path/to/data/public_datasets/ReXGroundingCT/segmentations")
DEFAULT_OUTPUT = Path("ReXGroundingCT_val")

SPLIT_ALIASES = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation"],
    "test": ["test", "testing"],
}
ID_FIELDS = ["name", "VolumeName", "volume_name", "volumeName", "image", "series_id", "id"]

# CT-RATE volume naming: <split>_<patient>_<scan>_<recon>, e.g. "train_13082_a_1"
STEM_RE = re.compile(r"^(?P<split>[A-Za-z]+)_(?P<patient>\d+)_(?P<scan>[A-Za-z0-9]+)_(?P<recon>\d+)$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> dict | list:
    with open(manifest_path) as f:
        return json.load(f)


def extract_split_records(manifest: dict | list, split_name: str, split_key: str | None) -> list[dict]:
    aliases = SPLIT_ALIASES[split_name]

    if isinstance(manifest, dict):
        if split_key:
            if split_key not in manifest:
                raise KeyError(f"'{split_key}' not in manifest top-level keys: {list(manifest.keys())}")
            records = manifest[split_key]
        else:
            matches = [k for k in manifest if k.lower() in aliases]
            if not matches:
                raise KeyError(
                    f"No {split_name}-like key found among manifest top-level keys: "
                    f"{list(manifest.keys())}. Use --split-key-{split_name} to specify one."
                )
            if len(matches) > 1:
                raise KeyError(
                    f"Multiple {split_name}-like keys found: {matches}. Use --split-key-{split_name} to disambiguate."
                )
            records = manifest[matches[0]]
            log.info("Auto-detected %s split key: %r", split_name, matches[0])
    elif isinstance(manifest, list):
        records = [r for r in manifest if str(r.get("split", "")).lower() in aliases]
        if not records:
            raise KeyError(f"Manifest is a list but no records have a 'split' field matching {split_name}.")
        log.info("Auto-detected list-style manifest filtered by 'split' field for %s", split_name)
    else:
        raise TypeError(f"Unsupported manifest top-level type: {type(manifest)}")

    if not isinstance(records, list):
        raise TypeError(f"Expected a list of records for the {split_name} split, got {type(records)}")
    return records


def extract_id(record: dict, id_field: str | None) -> str | None:
    if id_field:
        return record.get(id_field)
    for field in ID_FIELDS:
        if field in record and record[field]:
            return record[field]
    return None


def resolve_path(stem: str, dataset_root: Path) -> Path | None:
    """Construct a volume's expected path from its stem, without scanning any directory."""
    m = STEM_RE.match(stem)
    if not m:
        return None
    patient_dir = f"{m['split']}_{m['patient']}"
    scan_dir = f"{patient_dir}_{m['scan']}"
    return dataset_root / f"{m['split']}_fixed" / patient_dir / scan_dir / f"{stem}.nii.gz"


def strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name


def already_copied(src: Path, dst: Path) -> bool:
    """True if dst exists and matches src's size (guards against a prior interrupted copy)."""
    return dst.exists() and dst.stat().st_size == src.stat().st_size


def process_split(records: list[dict], args, seen_ids: set[str],
                   split_name: str) -> tuple[list[dict], int, int, int, int]:
    """Copy one split's images/masks; volumes already seen (in this or a prior split), or already
    present in the output at matching size, are skipped."""
    unmatched = []
    images_copied = 0
    masks_copied = 0
    images_skipped = 0
    masks_skipped = 0

    for record in records:
        vol_id = extract_id(record, args.id_field)
        if not vol_id:
            unmatched.append({"split": split_name, "reason": "no_id_field", "record": record})
            continue

        stem = strip_nii_ext(str(vol_id))

        if stem in seen_ids:
            continue
        seen_ids.add(stem)

        src = resolve_path(stem, args.dataset_root)
        if src is None:
            unmatched.append({"split": split_name, "kind": "image", "reason": "unparseable_name", "volume_id": vol_id})
        elif not src.exists():
            unmatched.append({"split": split_name, "kind": "image", "reason": "not_found",
                               "volume_id": vol_id, "expected_path": str(src)})
        else:
            rel = src.relative_to(args.dataset_root)
            dst = args.output / rel
            if already_copied(src, dst):
                images_skipped += 1
            elif args.dry_run:
                log.info("DRY   [%s] %s  ->  %s", split_name, src, dst)
                images_copied += 1
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                images_copied += 1

        # The challenge withholds test-split masks entirely, so don't bother looking for them.
        if not args.no_segmentations and split_name != "test":
            seg_src = args.segmentations_root / f"{stem}.nii.gz"
            if not seg_src.exists():
                unmatched.append({"split": split_name, "kind": "segmentation", "reason": "not_found",
                                   "volume_id": vol_id, "expected_path": str(seg_src)})
            else:
                seg_dst = args.output / "segmentations" / seg_src.name
                if already_copied(seg_src, seg_dst):
                    masks_skipped += 1
                elif args.dry_run:
                    log.info("DRY   [%s] %s  ->  %s", split_name, seg_src, seg_dst)
                    masks_copied += 1
                else:
                    seg_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(seg_src, seg_dst)
                    masks_copied += 1

    return unmatched, images_copied, masks_copied, images_skipped, masks_skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--dataset-root", type=Path, default=CT_RATE_DATASET_ROOT,
                         help="CT-RATE dataset root containing <split>_fixed subdirs")
    parser.add_argument("--segmentations-root", type=Path, default=SEGMENTATIONS_ROOT,
                         help="Flat directory of segmentation masks keyed by volume name")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"],
                         help="Which splits to copy (default: train val; pass 'test' explicitly to include it)")
    parser.add_argument("--split-key-train", type=str, default=None,
                         help="Override auto-detected manifest key for the train split")
    parser.add_argument("--split-key-val", type=str, default=None,
                         help="Override auto-detected manifest key for the val split")
    parser.add_argument("--split-key-test", type=str, default=None,
                         help="Override auto-detected manifest key for the test split")
    parser.add_argument("--id-field", type=str, default=None,
                         help="Override auto-detected record field naming the CT-RATE volume")
    parser.add_argument("--no-manifest", action="store_true",
                         help="Skip writing the filtered validation manifest into the output folder")
    parser.add_argument("--no-segmentations", action="store_true",
                         help="Skip copying segmentation masks")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without copying")
    args = parser.parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s", args.manifest)
        sys.exit(1)
    if not args.dataset_root.exists():
        log.error("CT-RATE dataset root not found: %s", args.dataset_root)
        sys.exit(1)
    if not args.no_segmentations and not args.segmentations_root.exists():
        log.error("Segmentations root not found: %s", args.segmentations_root)
        sys.exit(1)

    manifest = load_manifest(args.manifest)
    split_key_overrides = {"train": args.split_key_train, "val": args.split_key_val, "test": args.split_key_test}

    all_unmatched = []
    seen_ids = set()
    totals = {}

    for split_name in args.splits:
        records = extract_split_records(manifest, split_name, split_key_overrides[split_name])
        log.info("Loaded %d %s record(s) from %s", len(records), split_name, args.manifest)
        if records:
            log.info("Sample %s record keys: %s", split_name, list(records[0].keys()))

        unmatched, images_copied, masks_copied, images_skipped, masks_skipped = process_split(
            records, args, seen_ids, split_name)
        all_unmatched.extend(unmatched)
        totals[split_name] = (images_copied, masks_copied, images_skipped, masks_skipped)
        log.info("[%s] images copied=%d  masks copied=%d  images skipped(existing)=%d  masks skipped(existing)=%d  unmatched=%d",
                  split_name, images_copied, masks_copied, images_skipped, masks_skipped, len(unmatched))

        if not args.dry_run and not args.no_manifest:
            args.output.mkdir(parents=True, exist_ok=True)
            manifest_out = args.output / f"{split_name}_manifest.json"
            with open(manifest_out, "w") as f:
                json.dump(records, f, indent=2)
            log.info("Wrote filtered %s manifest (%d records) to %s", split_name, len(records), manifest_out)

    total_images = sum(images for images, _, _, _ in totals.values())
    total_masks = sum(masks for _, masks, _, _ in totals.values())
    total_images_skipped = sum(skipped for _, _, skipped, _ in totals.values())
    total_masks_skipped = sum(skipped for _, _, _, skipped in totals.values())

    log.info("-" * 60)
    log.info("Done.  unique volumes=%d  images copied=%d  masks copied=%d  "
              "images skipped(existing)=%d  masks skipped(existing)=%d  unmatched=%d",
              len(seen_ids), total_images, total_masks, total_images_skipped, total_masks_skipped, len(all_unmatched))

    if all_unmatched:
        log.warning("Sample of unmatched (up to 10): %s", all_unmatched[:10])

    if not args.dry_run and all_unmatched:
        args.output.mkdir(parents=True, exist_ok=True)
        unmatched_path = args.output / "unmatched.json"
        with open(unmatched_path, "w") as f:
            json.dump(all_unmatched, f, indent=2)
        log.info("Wrote %d unmatched record(s) to %s", len(all_unmatched), unmatched_path)


if __name__ == "__main__":
    main()
