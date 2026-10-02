#!/usr/bin/env python3
"""
Compute the physical volume (mL == cm^3, and raw mm^3) of every ground-truth
segmentation mask referenced by the official ED and oncology test manifests
(official_splits/ed_official_test_data.json, official_splits/onc_official_test_data.json).

Volume is voxel_count * voxel_volume, where voxel_volume comes from each
mask NIfTI's own affine (nib.get_zooms()) -- not a hardcoded spacing -- so
this works whether pointed at the resampled (1.5x1.5x3.0mm) or native-grid
mask directories. A voxel counts as foreground if value > 0.5 (masks are
binary 0/1 label volumes).

Mirrors scripts/evaluation/evaluate_{ed,onc}_official_test.py's conventions:
same manifests, same --mask-dir resolution, same ED "CATEGORY/accession/..."
category-from-path grouping and ONC "finding"-field grouping (falling back to
"unknown" for both a missing key and an explicit finding=null), same
sentence=null skip.

Also writes a 3-panel log-scale histogram (ED / oncology / combined) of each
sample's individual volume_mL to <output's parent>/volume_histograms.png.

Usage
-----
    python scripts/compute_mask_volumes.py \
        --ed-mask-dir /path/to/data/inhouse_abdominal_ct/ED_TEST_SET_resampled \
        --onc-mask-dir /path/to/data/inhouse_abdominal_ct/labels_resampled \
        --output outputs/mask_volumes/results.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ED_MANIFEST = REPO_ROOT / "official_splits" / "ed_official_test_data.json"
DEFAULT_ONC_MANIFEST = REPO_ROOT / "official_splits" / "onc_official_test_data.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ed-manifest", type=Path, default=DEFAULT_ED_MANIFEST)
    parser.add_argument("--onc-manifest", type=Path, default=DEFAULT_ONC_MANIFEST)
    parser.add_argument("--ed-mask-dir", required=True,
                         help="Base dir ED manifest's relative mask paths resolve against "
                              "(e.g. .../inhouse_abdominal_ct/ED_TEST_SET_resampled)")
    parser.add_argument("--onc-mask-dir", required=True,
                         help="Base dir ONC manifest's relative mask paths resolve against "
                              "(e.g. .../inhouse_abdominal_ct/labels_resampled)")
    parser.add_argument("--output", default="outputs/mask_volumes/results.json",
                         help="Where to write the full per-sample + summary JSON")
    return parser.parse_args()


def category_of(mask_id: str) -> str:
    """ED masks are laid out CATEGORY/accession/Struct_....nii.gz."""
    return mask_id.split("/")[0]


def mask_volume_mm3(mask_path: Path) -> tuple[float, int]:
    """Returns (volume_mm3, foreground_voxel_count) for a binary label NIfTI."""
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    voxel_vol_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    n_fg = int(np.sum(data > 0.5))
    return n_fg * voxel_vol_mm3, n_fg


def process_split(split_name: str, manifest_path: Path, mask_dir: Path, group_fn) -> list[dict]:
    with open(manifest_path) as f:
        samples = json.load(f)

    records = []
    n_missing = 0
    for sample in samples:
        mask_id = sample["mask"]
        mask_path = mask_dir / mask_id
        if not mask_path.exists():
            logger.warning(f"[{split_name}] missing mask, skipping: {mask_path}")
            n_missing += 1
            continue
        try:
            volume_mm3, n_voxels = mask_volume_mm3(mask_path)
        except Exception:
            logger.warning(f"[{split_name}] failed to load mask, skipping: {mask_path}", exc_info=True)
            n_missing += 1
            continue
        records.append({
            "id": mask_id,
            "group": group_fn(sample),
            "voxel_count": n_voxels,
            "volume_mm3": volume_mm3,
            "volume_mL": volume_mm3 / 1000.0,
        })

    logger.info(f"[{split_name}] computed volumes for {len(records)}/{len(samples)} samples "
                f"({n_missing} missing/failed)")
    return records


def summarize(records: list[dict]) -> dict:
    vols = [r["volume_mL"] for r in records]
    return {
        "n_samples": len(records),
        "volume_mL_mean": float(np.mean(vols)),
        "volume_mL_std": float(np.std(vols)),
        "volume_mL_median": float(np.median(vols)),
        "volume_mL_min": float(np.min(vols)),
        "volume_mL_max": float(np.max(vols)),
    }


def plot_volume_histograms(ed_records: list[dict], onc_records: list[dict], output_path: Path) -> None:
    """3-panel log-scale histogram of each sample's individual volume_mL (ED, ONC, combined)."""
    panels = (
        ("ED", [r["volume_mL"] for r in ed_records]),
        ("Oncology", [r["volume_mL"] for r in onc_records]),
        ("Combined", [r["volume_mL"] for r in ed_records] + [r["volume_mL"] for r in onc_records]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, vols) in zip(axes, panels):
        if not vols:
            ax.set_title(f"{title} (n=0)")
            continue
        # log-scale x needs strictly positive values -- floor zero/near-zero
        # volumes to a small epsilon so empty masks still show at the left edge
        floored = np.clip(vols, 1e-2, None)
        if floored.max() > floored.min():
            bins = np.logspace(np.log10(floored.min()), np.log10(floored.max()), 50)
        else:
            bins = np.logspace(np.log10(floored.min()) - 0.1, np.log10(floored.min()) + 0.1, 2)
        ax.hist(floored, bins=bins, color="steelblue", edgecolor="black", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Volume (mL, log scale)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} (n={len(vols)})")

    fig.suptitle("Ground-truth mask volume per sample")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Wrote volume histograms to {output_path}")


def main() -> None:
    args = parse_args()

    ed_records = process_split("ED", args.ed_manifest, Path(args.ed_mask_dir), lambda s: category_of(s["mask"]))
    onc_records = process_split("ONC", args.onc_manifest, Path(args.onc_mask_dir), lambda s: s.get("finding") or "unknown")

    results = {}
    for split_name, records in (("ed", ed_records), ("onc", onc_records)):
        by_group = defaultdict(list)
        for r in records:
            by_group[r["group"]].append(r)
        results[split_name] = {
            "overall": summarize(records),
            "by_group": {g: summarize(recs) for g, recs in sorted(by_group.items())},
            "samples": records,
        }
        logger.info(f"[{split_name.upper()}] overall: {results[split_name]['overall']}")
        for g, s in results[split_name]["by_group"].items():
            logger.info(f"  {g}: n={s['n_samples']} mean={s['volume_mL_mean']:.2f}mL "
                        f"median={s['volume_mL_median']:.2f}mL")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote results to {output_path}")

    plot_volume_histograms(ed_records, onc_records, output_path.parent / "volume_histograms.png")


if __name__ == "__main__":
    main()
