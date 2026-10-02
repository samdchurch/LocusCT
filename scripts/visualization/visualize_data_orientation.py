#!/usr/bin/env python3
"""
Visualize input CT + GT mask orientation across all three data sources this
project trains/evaluates on, to sanity-check that GrounderDataset's
canonicalization (load_nifti_canonical: nib.as_closest_canonical -> (D, H, W))
produces consistently-oriented volumes no matter which preprocessing pipeline
produced the underlying files:

  train    whatever data.train_manifest currently points at (default
           official_splits/all_data_train.json) -- images/masks produced by
           resample_and_crop.py + resample_masks.py.
  ed_test  official_splits/ed_official_test_data.json -- same image pipeline,
           but masks resampled from the newer ED_TEST_SET/ tree.
  rex      official_splits/ReXGroundingCT_val.json -- images/masks produced by
           resample_rexgroundingct.py/resample_rexgroundingct_val.py's separate,
           manual per-finding resampling (see that script's docstring for why it
           can't use resample_masks.py's resample_from_to approach).

No model/checkpoint is involved -- each sample is loaded through GrounderDataset
exactly the way training/eval do, then rendered as image + GT mask
(axial/coronal/sagittal, green contour, matching visualize.py's style) so a human
can visually confirm nothing is flipped/rotated relative to the others. Each
dataset's embedding_cache is force-disabled (live tokenization instead) since
mask orientation has nothing to do with how the text side is encoded, and this
keeps the script runnable standalone without requiring any of the three manifests
to already be covered by a precomputed embedding cache.

Displayed with a fixed --window-hu-min/--window-hu-max HU window (default -100 to
200, a standard soft-tissue window), converted into the same normalized space
_apply_windows produced using data.hu_min/hu_max from the config -- NOT the raw
[-1, 1]-normalized array values directly. Each sample's raw min/max is logged
alongside so a real normalization bug (values outside what _apply_windows should
have produced) is visible in the log rather than just showing up as a
posterized-looking image.

Usage
-----
    python visualize_data_orientation.py --config configs/default.yaml
    python visualize_data_orientation.py --config configs/default.yaml --n-samples 8 --output-dir orientation_check
    python visualize_data_orientation.py --config configs/default.yaml \
        --ed-image-dir /path/to/data/inhouse_abdominal_ct/nifti_resampled \
        --ed-mask-dir /path/to/data/inhouse_abdominal_ct/ED_TEST_SET_resampled \
        --override model.text_encoder_name=/path/to/local/Qwen3-Embedding-8B
"""

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import GrounderDataset
from train import apply_overrides

DEFAULT_ED_MANIFEST = "official_splits/ed_official_test_data.json"
DEFAULT_ED_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti_resampled"
DEFAULT_ED_MASK_DIR = "/path/to/data/inhouse_abdominal_ct/ED_TEST_SET_resampled"

DEFAULT_REX_MANIFEST = "official_splits/ReXGroundingCT_val.json"
DEFAULT_REX_DATA_ROOT = "/path/to/data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-samples", type=int, default=5, help="Samples to render per dataset")
    parser.add_argument("--output-dir", default="outputs/viz/orientation_check")

    parser.add_argument("--train-manifest", default=None,
                         help="Default: data.train_manifest from config (first entry if it's a list)")

    parser.add_argument("--ed-manifest", default=DEFAULT_ED_MANIFEST)
    parser.add_argument("--ed-image-dir", default=DEFAULT_ED_IMAGE_DIR)
    parser.add_argument("--ed-mask-dir", default=DEFAULT_ED_MASK_DIR)

    parser.add_argument("--rex-manifest", default=DEFAULT_REX_MANIFEST)
    parser.add_argument("--rex-data-root", default=DEFAULT_REX_DATA_ROOT)

    parser.add_argument("--window-hu-min", type=float, default=-100.0,
                         help="Display window lower bound in HU (default: -100)")
    parser.add_argument("--window-hu-max", type=float, default=200.0,
                         help="Display window upper bound in HU (default: 200)")

    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model",
    )
    return parser.parse_args()


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    """(D, H, W) binary mask -> (d, h, w) centroid; falls back to volume center."""
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    d = int(np.clip(d, 0, mask.shape[0] - 1))
    h = int(np.clip(h, 0, mask.shape[1] - 1))
    w = int(np.clip(w, 0, mask.shape[2] - 1))
    return d, h, w


def _trim_volume(vol: np.ndarray, pad_amounts: torch.Tensor) -> np.ndarray:
    """Remove end-padding added by GrounderDataset's fixed spatial_mode."""
    pad_D, pad_H, pad_W = (int(p) for p in pad_amounts)
    D = vol.shape[0] - pad_D if pad_D > 0 else vol.shape[0]
    H = vol.shape[1] - pad_H if pad_H > 0 else vol.shape[1]
    W = vol.shape[2] - pad_W if pad_W > 0 else vol.shape[2]
    return vol[:D, :H, :W]


def _safe_filename(sample_id: str) -> str:
    name = sample_id.replace("/", "_").replace("\\", "_")
    for ext in (".nii.gz", ".nii", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name[-150:] if len(name) > 150 else name


def _save_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    title: str,
    out_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    """Axial/coronal/sagittal slices centered on the GT mask centroid, GT contour in green.
    No prediction -- this is a pre-model data sanity check, not an eval visualization."""
    d, h, w = _mask_centroid(gt_mask)

    views = [
        ("Axial",    image[d, :, :], gt_mask[d, :, :]),
        ("Coronal",  image[:, h, :], gt_mask[:, h, :]),
        ("Sagittal", image[:, :, w], gt_mask[:, :, w]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    fig.patch.set_facecolor("black")

    for ax, (label, img_sl, gt_sl) in zip(axes, views):
        img_sl = np.rot90(img_sl, 2)
        gt_sl = np.rot90(gt_sl, 2)
        img_sl = np.clip(img_sl, vmin, vmax)
        img_sl = (img_sl - vmin) / (vmax - vmin)
        if label == "Axial":
            logger.info(f"  imshow input: dtype={img_sl.dtype}  shape={img_sl.shape}  "
                        f"contiguous={img_sl.flags['C_CONTIGUOUS']}  "
                        f"min={img_sl.min():.4f}  max={img_sl.max():.4f}  "
                        f"unique_vals<=5={len(np.unique(img_sl)) <= 5}")
        ax.imshow(img_sl, cmap="gray", aspect="equal", origin="upper")
        if gt_sl.any():
            ax.contour(gt_sl, levels=[0.5], colors=["#00e676"], linewidths=1.5)
        ax.set_title(label, color="white", fontsize=11, pad=4)
        ax.set_facecolor("black")
        ax.axis("off")

    fig.suptitle(textwrap.fill(title, width=100), color="white", fontsize=9, y=1.03, va="bottom")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def visualize_dataset(
    name: str,
    manifest_path: str,
    tokenizer_name: str,
    cfg: dict,
    image_dir: str,
    mask_dir: str,
    n_samples: int,
    output_dir: Path,
    window_hu_min: float,
    window_hu_max: float,
) -> None:
    logger.info(f"[{name}] manifest={manifest_path}  image_dir={image_dir}  mask_dir={mask_dir}")

    dataset = GrounderDataset(
        manifest_path=manifest_path,
        tokenizer_name=tokenizer_name,
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        max_text_len=cfg["data"]["max_text_len"],
        hu_min=cfg["data"]["hu_min"],
        hu_max=cfg["data"]["hu_max"],
        multi_window=cfg["data"].get("multi_window", False),
        spatial_mode=cfg["data"]["spatial_mode"],
        augment=False,
        image_dir=image_dir,
        mask_dir=mask_dir,
        embedding_cache="",  # force live tokenization -- irrelevant to image/mask orientation
        max_samples=n_samples,
    )

    if len(dataset) == 0:
        logger.warning(f"[{name}] no samples found (check paths above) -- skipping")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)
    sentence_lookup = {s["mask"]: s.get("sentence") for s in manifest}

    # Same HU -> [-1, 1] mapping _apply_windows used, applied to the requested
    # display bounds instead of the full hu_min/hu_max clip range.
    hu_min, hu_max = cfg["data"]["hu_min"], cfg["data"]["hu_max"]
    def _hu_to_norm(hu: float) -> float:
        return (hu - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
    vmin, vmax = _hu_to_norm(window_hu_min), _hu_to_norm(window_hu_max)
    logger.info(f"[{name}] display window {window_hu_min:.0f}..{window_hu_max:.0f} HU "
                f"-> normalized [{vmin:.3f}, {vmax:.3f}] (using data.hu_min={hu_min}, data.hu_max={hu_max})")

    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(n_samples, len(dataset))
    for i in tqdm(range(n), desc=f"Rendering {name}"):
        item = dataset[i]
        sample_id = item["id"]

        # .float() explicit to match visualize.py's extraction exactly
        # (image_t[0, 0].cpu().float().numpy()), rather than relying on the
        # tensor already being float32 by construction.
        image_np = _trim_volume(item["image"][0].float().numpy(), item["pad_amounts"])
        gt_np = _trim_volume(item["mask"][0].float().numpy(), item["pad_amounts"])
        sentence = sentence_lookup.get(sample_id, sample_id)

        # min/max alone isn't useful here: any real volume with some background/air
        # (below -1000 HU) and some bone/contrast/artifact (above 1000 HU) anywhere
        # in it will legitimately clip to exactly [-1, 1] -- that's expected, not a
        # bug. Percentiles show the bulk of the distribution, which is what actually
        # determines whether the display window covers real tissue or not.
        pct = np.percentile(image_np, [1, 5, 50, 95, 99])
        logger.info(f"[{name}] {sample_id}: range [{image_np.min():.3f}, {image_np.max():.3f}]  "
                    f"p1={pct[0]:.3f} p5={pct[1]:.3f} p50={pct[2]:.3f} p95={pct[3]:.3f} p99={pct[4]:.3f}  "
                    f"(display window [{vmin:.3f}, {vmax:.3f}])")

        title = f"[{name}] {sample_id}\n{sentence}"
        out_path = out_dir / f"{_safe_filename(sample_id)}.png"
        _save_figure(image_np, gt_np, title, out_path, vmin=vmin, vmax=vmax)

    logger.info(f"[{name}] wrote {n} visualization(s) -> {out_dir}")


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    tokenizer_name = cfg["model"]["text_encoder_name"]
    output_dir = Path(args.output_dir)

    train_manifest = args.train_manifest
    if train_manifest is None:
        tm = cfg["data"]["train_manifest"]
        first = tm[0] if isinstance(tm, list) else tm
        if isinstance(first, dict):
            train_manifest = first["manifest"]
            if "image_dir" in first or "mask_dir" in first:
                logger.warning(
                    f"train_manifest entry {train_manifest!r} has its own image_dir/mask_dir "
                    f"override; this script uses the top-level data.image_dir/mask_dir instead"
                )
        else:
            train_manifest = first

    datasets = [
        dict(
            name="train",
            manifest_path=train_manifest,
            image_dir=cfg["data"]["image_dir"],
            mask_dir=cfg["data"]["mask_dir"],
        ),
        dict(
            name="ed_test",
            manifest_path=args.ed_manifest,
            image_dir=args.ed_image_dir,
            mask_dir=args.ed_mask_dir,
        ),
        dict(
            name="rex",
            manifest_path=args.rex_manifest,
            image_dir=args.rex_data_root,
            mask_dir=args.rex_data_root,
        ),
    ]

    for ds in datasets:
        visualize_dataset(
            cfg=cfg,
            tokenizer_name=tokenizer_name,
            n_samples=args.n_samples,
            output_dir=output_dir,
            window_hu_min=args.window_hu_min,
            window_hu_max=args.window_hu_max,
            **ds,
        )

    logger.info(f"Done. Compare {output_dir}/train/, {output_dir}/ed_test/, {output_dir}/rex/ "
                f"side by side -- orientation should look consistent across all three.")


if __name__ == "__main__":
    main()
