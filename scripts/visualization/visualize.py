"""
Visualize Grounder predictions on a validation split.

Usage:
    python visualize.py --config configs/default.yaml \
        --run_dir runs/h200/ch16_bs4_lr1e-03_fixed352x352x192_cross_attention \
        [--n_samples 300] [--split val] [--output_dir path/to/out]

Outputs one PNG per sample in <run_dir>/viz/ (or --output_dir).
Each image shows axial / coronal / sagittal slices centered on the GT
mask centroid, with green GT contour and red predicted contour overlaid.
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
from matplotlib.lines import Line2D
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import build_dataloader
from models.grounder import Grounder
from training.trainer import _text_kwargs, _trim_padding
from utils.metrics import dice_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Grounder predictions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True,
                        help="Run directory, e.g. runs/h200/ch16_bs4_lr1e-03_fixed352x352x192_cross_attention")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--output_dir", default=None,
                        help="Where to write PNGs. Default: <run_dir>/viz")
    return parser.parse_args()


def _build_sentence_lookup(manifest_path: str) -> dict[str, str]:
    with open(manifest_path) as f:
        samples = json.load(f)
    return {s["mask"]: s["sentence"] for s in samples}


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    """(D, H, W) binary mask → (d, h, w) centroid; falls back to volume center."""
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    d = int(np.clip(d, 0, mask.shape[0] - 1))
    h = int(np.clip(h, 0, mask.shape[1] - 1))
    w = int(np.clip(w, 0, mask.shape[2] - 1))
    return d, h, w


def _trim_volume(vol: np.ndarray, pad_amounts: torch.Tensor) -> np.ndarray:
    """Remove end-padding from a (D, H, W) numpy array."""
    pad_D, pad_H, pad_W = (int(p) for p in pad_amounts)
    D = vol.shape[0] - pad_D if pad_D > 0 else vol.shape[0]
    H = vol.shape[1] - pad_H if pad_H > 0 else vol.shape[1]
    W = vol.shape[2] - pad_W if pad_W > 0 else vol.shape[2]
    return vol[:D, :H, :W]


def _save_figure(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    sentence: str,
    dice: float,
    out_path: Path,
    vmin: float = -0.15,
    vmax: float = 0.25,
) -> None:
    d, h, w = _mask_centroid(gt_mask)

    def _slices(ci, cj, ck):
        return [
            ("Axial",    image[ci, :, :],  gt_mask[ci, :, :],  pred_mask[ci, :, :]),
            ("Coronal",  image[:, cj, :],  gt_mask[:, cj, :],  pred_mask[:, cj, :]),
            ("Sagittal", image[:, :, ck],  gt_mask[:, :, ck],  pred_mask[:, :, ck]),
        ]

    rows_views = [_slices(d, h, w)]
    row_labels = ["GT center"]

    pred_in_gt_slices = (
        pred_mask[d, :, :].any() or
        pred_mask[:, h, :].any() or
        pred_mask[:, :, w].any()
    )
    if not pred_in_gt_slices and pred_mask.any():
        pd, ph, pw = _mask_centroid(pred_mask)
        rows_views.append(_slices(pd, ph, pw))
        row_labels.append("Pred center")

    nrows = len(rows_views)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 5 * nrows), facecolor="black")
    fig.patch.set_facecolor("black")

    if nrows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (views, row_label) in enumerate(zip(rows_views, row_labels)):
        for col_idx, (label, img_sl, gt_sl, pred_sl) in enumerate(views):
            ax = axes[row_idx, col_idx]
            img_sl = np.rot90(img_sl, 2)
            gt_sl = np.rot90(gt_sl, 2)
            pred_sl = np.rot90(pred_sl, 2)
            ax.imshow(img_sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal", origin="upper")
            if gt_sl.any():
                ax.contour(gt_sl, levels=[0.5], colors=["#00e676"], linewidths=1.5)
            if pred_sl.any():
                ax.contour(pred_sl, levels=[0.5], colors=["#ff1744"], linewidths=1.5)
            ax.set_title(label, color="white", fontsize=11, pad=4)
            ax.set_facecolor("black")
            ax.axis("off")
            if col_idx == 0 and nrows > 1:
                ax.text(0.02, 0.97, row_label, transform=ax.transAxes,
                        color="white", fontsize=9, va="top", ha="left",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    legend = [
        Line2D([0], [0], color="#00e676", linewidth=1.5, label="GT"),
        Line2D([0], [0], color="#ff1744", linewidth=1.5, label="Pred"),
    ]
    axes[-1, -1].legend(handles=legend, loc="lower right", fontsize=9,
                        framealpha=0.4, facecolor="black", labelcolor="white",
                        edgecolor="gray")

    title = textwrap.fill(sentence, width=90) + f"\nDice: {dice:.3f}"
    fig.suptitle(title, color="white", fontsize=9, y=1.03, va="bottom")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def _safe_filename(sample_id: str) -> str:
    name = sample_id.replace("/", "_").replace("\\", "_")
    for ext in (".nii.gz", ".nii", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name[-150:] if len(name) > 150 else name


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = cfg["inference"].get("threshold", 0.5)

    # Convert HU display window to normalized [-1, 1] space
    hu_min = cfg["data"]["hu_min"]
    hu_max = cfg["data"]["hu_max"]
    def _hu_to_norm(hu: float) -> float:
        return (hu - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
    vmin = _hu_to_norm(-150)
    vmax = _hu_to_norm(250)

    model = Grounder(
        text_encoder_name=cfg["model"]["text_encoder_name"],
        text_proj_dim=cfg["model"]["text_proj_dim"],
        freeze_text_encoder=True,
        finetune_last_n_layers=0,
        unet_base_channels=cfg["model"]["unet_base_channels"],
        unet_channel_mult=cfg["model"]["unet_channel_mult"],
        num_heads=cfg["model"]["num_heads"],
        target_q_tokens=cfg["model"]["target_q_tokens"],
        dropout=0.0,
        in_channels=3 if cfg["data"].get("multi_window") else 1,
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        load_text_backbone=not cfg["data"].get("embedding_cache"),
        fusion_type=cfg["model"].get("fusion_type", "cross_attention"),
        voxtell_guidance_dim=cfg["model"].get("voxtell", {}).get("guidance_dim", 32),
        voxtell_prompt_decoder_dim=cfg["model"].get("voxtell", {}).get("prompt_decoder_dim", 256),
        voxtell_prompt_decoder_layers=cfg["model"].get("voxtell", {}).get("prompt_decoder_layers", 6),
        voxtell_prompt_decoder_heads=cfg["model"].get("voxtell", {}).get("prompt_decoder_heads", 8),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(
        f"Loaded epoch {ckpt.get('epoch', '?')}  "
        f"best_dice={ckpt.get('best_dice', float('nan')):.4f}"
    )

    manifest_key = f"{args.split}_manifest"
    manifest_path = cfg["data"][manifest_key]
    sentence_lookup = _build_sentence_lookup(manifest_path)

    # Shallow-copy cfg so we can override max_samples and batch_size without side effects
    viz_cfg = {**cfg, "data": {**cfg["data"], "max_samples": args.n_samples},
               "training": {**cfg["training"], "batch_size": 1}}

    loader = build_dataloader(manifest_path, viz_cfg, split=args.split, num_workers=4)
    logger.info(f"Visualizing up to {args.n_samples} {args.split} samples → {output_dir}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Visualizing"):
            image_t = batch["image"].to(device)
            mask_t = batch["mask"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(image_t, **_text_kwargs(batch, device))

            logits, mask_t = _trim_padding(logits, mask_t, batch["pad_amounts"])

            sample_id = batch["id"][0]
            dice = dice_score(logits, mask_t, threshold=threshold, from_logits=True)[0].item()

            image_np = _trim_volume(image_t[0, 0].cpu().float().numpy(), batch["pad_amounts"][0])
            gt_np    = mask_t[0, 0].cpu().float().numpy()
            pred_np  = (torch.sigmoid(logits[0, 0]).cpu().float().numpy() > threshold).astype(np.float32)

            sentence = sentence_lookup.get(sample_id, sample_id)
            out_path = output_dir / f"{_safe_filename(sample_id)}.png"
            _save_figure(image_np, gt_np, pred_np, sentence, dice, out_path, vmin=vmin, vmax=vmax)

    logger.info(f"Done. {len(loader)} visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
