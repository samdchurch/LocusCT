#!/usr/bin/env python3
"""
Score the published SAT-Pro checkpoint (github.com/zhaoziheng/SAT) on the official
oncology held-out test set (official_splits/onc_official_test_data.json), reporting
Dice and hit rate (dice >= 0.1) overall and per finding. Pass --visualize to also write
one GT/pred overlay PNG per case (axial/coronal/sagittal, green GT contour, red
predicted contour) to <output's parent>/viz/<finding>/, matching evaluate_voxtell_onc.py
and evaluate_biomedparse_onc.py's style.

Like the other two baselines' eval scripts, this runs on native-resolution NIfTI
volumes and isn't a strictly apples-to-apples comparison against this repo's own
352x352x180-resampled-grid eval scripts.

Model: SAT-Pro, run via SAT's own real `evaluate.inference_engine.inference()` --
loaded and called exactly as sat/SAT/inference.py's own main() does (UNET-L vision
backbone, 'ours' text encoder built on BioLORD, both DDP-wrapped -- SAT's
build_maskformer()/Text_Encoder() do this unconditionally, so this script (like
inference.py itself) must be launched via `torchrun --nproc_per_node=1`, even for a
single GPU). Preprocessing (RAS orientation, (1,1,3)mm spacing, foreground crop, CT
HU-clip-then-z-score) is entirely SAT's own (data/inference_dataset.py), untouched
here. All findings for one scan are sent to SAT together as one multi-label query list
-- unlike BiomedParse's argmax-merged multi-prompt batching, SAT's queries produce
independent per-label sigmoid masks (see evaluate/inference_engine.py's own inference()
loop), so batching doesn't distort per-finding Dice the way it would for BiomedParse.

Two deliberate accommodations, neither of which touches SAT's own model/preprocessing
code:

1. SAT's inference() names each output file literally after the query text (fine for
   its own demo's short anatomical labels like "liver", but this dataset's referring
   expressions are full free-text report sentences up to 500+ characters -- well past
   most filesystems' ~255-byte filename limit, which would crash inference() outright
   via nib.save()). Patched at the nibabel level (see _patch_nib_save_for_long_labels)
   to substitute a length-safe, collision-resistant filename for any over-length label
   -- the text actually sent to the text encoder is never touched.

2. SAT's own sample-id derivation (image path's basename minus ".nii.gz") collides
   across different accessions in this dataset, whose image filenames are generic
   per-series names (e.g. "5__ST_W_C.nii.gz") repeated across many patients. Worked
   around by setting each jsonl line's "dataset" field to that image's own accession
   id, which SAT uses as the output directory's top-level folder -- no code change
   needed, just per-image input construction.

Also: SAT's inference() saves predictions with an identity affine (no spatial
registration back to the original image -- see evaluate/inference_engine.py), on the
resampled/cropped grid its own preprocessing produced. To score against this repo's
native-resolution GT masks, load_gt_mask_on_sat_grid() below replicates SAT's own
image transform pipeline for the mask (nearest-neighbor in place of bilinear), landing
it on the identical grid.

Usage
-----
    torchrun --nproc_per_node=1 evaluate_sat_onc.py \\
        --checkpoint /path/to/SAT_Pro.pth --text-encoder-checkpoint /path/to/text_encoder.pth
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "onc_official_test_data.json"
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"
DEFAULT_MASK_DIR = "/path/to/data/inhouse_abdominal_ct/ALL_LABELS"
DEFAULT_SAT_REPO = Path(__file__).resolve().parents[2] / "sat" / "SAT"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def str2bool(v: str) -> bool:
    return v.lower() in ("true", "t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                         help="Base dir the manifest's relative 'image' paths resolve against")
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR,
                         help="Base dir the manifest's relative 'mask' paths resolve against")
    parser.add_argument("--sat-repo", type=Path, default=DEFAULT_SAT_REPO,
                         help="Path to the cloned SAT source repo (has data/, model/, evaluate/, train/)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SAT_Pro.pth (vision backbone + decoder)")
    parser.add_argument("--text-encoder-checkpoint", type=Path, required=True, help="text_encoder.pth")
    parser.add_argument("--vision-backbone", default="UNET-L", help="UNET, UNET-L, UMamba, or SwinUNETR")
    parser.add_argument("--text-encoder", default="ours", help="ours, medcpt, or basebert")
    parser.add_argument("--crop-size", type=int, nargs="+", default=[288, 288, 96])
    parser.add_argument("--patch-size", type=int, nargs="+", default=[32, 32, 32])
    parser.add_argument("--deep-supervision", type=str2bool, default=False)
    parser.add_argument("--partial-load", type=str2bool, default=True)
    parser.add_argument("--text-encoder-partial-load", type=str2bool, default=True)
    parser.add_argument("--max-queries", type=int, default=256)
    parser.add_argument("--batchsize-3d", type=int, default=2)
    parser.add_argument("--pin-memory", type=str2bool, default=False)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rcd-dir", default=None,
                         help="Where SAT's own inference() writes its raw nifti output. Default: <output's parent>/sat_raw")
    parser.add_argument("--output", default="outputs/eval/sat_onc/results.json")
    parser.add_argument("--hit-threshold", type=float, default=0.1,
                         help="Dice value at/above which a sample counts as a 'hit'")
    parser.add_argument("--visualize", dest="visualize", action="store_true", default=False,
                         help="Write per-case GT/pred overlay PNGs (off by default)")
    parser.add_argument("--viz-output-dir", default=None,
                         help="Where to write visualization PNGs. Default: <output's parent>/viz")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Filename safety (see module docstring, deviation 1)
# ---------------------------------------------------------------------------

def _safe_label_filename(label: str) -> str:
    name = f"{label}.nii.gz"
    if len(name.encode("utf-8")) <= 200:
        return name
    digest = hashlib.sha1(label.encode("utf-8")).hexdigest()[:16]
    return f"{label[:150]}__{digest}.nii.gz"


def _patch_nib_save_for_long_labels() -> None:
    original_save = nib.save

    def patched_save(img, filename, **kwargs):
        filename = str(filename)
        directory, base = os.path.split(filename)
        if base.endswith(".nii.gz"):
            safe_base = _safe_label_filename(base[: -len(".nii.gz")])
            if safe_base != base:
                filename = os.path.join(directory, safe_base)
        return original_save(img, filename, **kwargs)

    nib.save = patched_save


def sat_sample_id(image_path: str) -> str:
    """Mirrors data/inference_dataset.py::load_image's own sample-id derivation."""
    return image_path.split("/")[-1].replace(".nii.gz", "")


def sat_output_path(rcd_dir: Path, accession: str, sample_id: str, sentence: str) -> Path:
    return Path(rcd_dir) / accession / f"seg_{sample_id}" / _safe_label_filename(sentence)


# ---------------------------------------------------------------------------
# Query jsonl construction (see module docstring, deviation 2)
# ---------------------------------------------------------------------------

def build_query_jsonl(kept_samples: list[dict], image_dir: str, jsonl_path: Path) -> list[dict]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for s in kept_samples:
        by_image[s["image"]].append(s)

    entries = []
    with open(jsonl_path, "w") as f:
        for image_rel, group in sorted(by_image.items()):
            accession = Path(image_rel).parts[-2]
            sentence_groups: dict[str, list[dict]] = defaultdict(list)
            for s in group:
                sentence_groups[s["sentence"].strip()].append(s)
            sentences = sorted(sentence_groups)
            f.write(json.dumps({
                "image": f"{image_dir}/{image_rel}",
                "label": sentences,
                "dataset": accession,
                "modality": "ct",
            }) + "\n")
            entries.append({"accession": accession, "image_rel": image_rel, "sentence_groups": sentence_groups})
    return entries


def run_sat_inference(args: argparse.Namespace, jsonl_path: Path, rcd_dir: Path) -> None:
    """Replicates sat/SAT/inference.py's own main() verbatim: same DDP bootstrap,
    same dataset/model/text-encoder construction, then hands off to SAT's own
    evaluate.inference_engine.inference() unmodified."""
    sys.path.insert(0, str(args.sat_repo.resolve()))
    # SAT's model/maskformer.py imports a customized dynamic_network_architectures
    # fork (PlainConvUNet with no num_classes arg -- a backbone-only variant, unlike
    # every public PyPI release of this package) vendored at
    # model/dynamic-network-architectures-main; see that directory's own setup.py.
    # It's not pip-installed into the container image, so it must be made importable
    # here, ahead of anything in site-packages.
    sys.path.insert(0, str((args.sat_repo / "model" / "dynamic-network-architectures-main").resolve()))
    from data.inference_dataset import Inference_Dataset, collate_fn
    from evaluate.inference_engine import inference as sat_inference
    from model.build_model import build_maskformer, load_checkpoint
    from model.text_encoder import Text_Encoder

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=7200))

    rcd_dir.mkdir(exist_ok=True, parents=True)

    testset = Inference_Dataset(str(jsonl_path), args.max_queries, args.batchsize_3d)
    sampler = DistributedSampler(testset)
    testloader = DataLoader(testset, sampler=sampler, batch_size=1, pin_memory=args.pin_memory,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    sampler.set_epoch(0)

    model = build_maskformer(args, device, gpu_id)
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=str(args.text_encoder_checkpoint),
        partial_load=args.text_encoder_partial_load,
        open_bert_layer=12,
        open_modality_embed=False,
        gpu_id=gpu_id,
        device=device,
    )
    model, _, _ = load_checkpoint(
        checkpoint=str(args.checkpoint),
        resume=False,
        partial_load=args.partial_load,
        model=model,
        device=device,
    )

    logger.info(f"Running SAT inference -> {rcd_dir}")
    sat_inference(model=model, text_encoder=text_encoder, device=device,
                  testset=testset, testloader=testloader, nib_dir=str(rcd_dir))


# ---------------------------------------------------------------------------
# GT alignment onto SAT's own preprocessing grid (see module docstring)
# ---------------------------------------------------------------------------

_GT_TRANSFORM = monai.transforms.Compose([
    monai.transforms.LoadImaged(keys=["image", "mask"]),
    monai.transforms.EnsureChannelFirstd(keys=["image", "mask"]),
    monai.transforms.Orientationd(axcodes="RAS", keys=["image", "mask"]),
    monai.transforms.Spacingd(keys=["image", "mask"], pixdim=(1, 1, 3), mode=("bilinear", "nearest")),
    monai.transforms.CropForegroundd(keys=["image", "mask"], source_key="image"),
])


def load_gt_mask_on_sat_grid(image_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    d = _GT_TRANSFORM({"image": image_path, "mask": mask_path})
    image = np.asarray(d["image"][0])
    gt = (np.asarray(d["mask"][0]) > 0.5).astype(np.float32)
    return image, gt


def dice_score_np(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5, smooth: float = 1.0) -> float:
    pred = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)
    intersection = (pred * target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    d = int(np.clip(d, 0, mask.shape[0] - 1))
    h = int(np.clip(h, 0, mask.shape[1] - 1))
    w = int(np.clip(w, 0, mask.shape[2] - 1))
    return d, h, w


def _safe_filename(sample_id: str) -> str:
    name = sample_id.replace("/", "_").replace("\\", "_")
    for ext in (".nii.gz", ".nii", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name[-150:] if len(name) > 150 else name


def _save_figure(
    image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, sentence: str, dice: float,
    out_path: Path, vmin: float = -150.0, vmax: float = 250.0,
) -> None:
    d, h, w = _mask_centroid(gt_mask)

    def _slices(ci, cj, ck):
        return [
            ("Axial",    image[ci, :, :], gt_mask[ci, :, :], pred_mask[ci, :, :]),
            ("Coronal",  image[:, cj, :], gt_mask[:, cj, :], pred_mask[:, cj, :]),
            ("Sagittal", image[:, :, ck], gt_mask[:, :, ck], pred_mask[:, :, ck]),
        ]

    rows_views = [_slices(d, h, w)]
    row_labels = ["GT center"]

    pred_in_gt_slices = (
        pred_mask[d, :, :].any() or pred_mask[:, h, :].any() or pred_mask[:, :, w].any()
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
                        framealpha=0.4, facecolor="black", labelcolor="white", edgecolor="gray")

    title = textwrap.fill(sentence, width=90) + f"\nDice: {dice:.3f}"
    fig.suptitle(title, color="white", fontsize=9, y=1.03, va="bottom")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def summarize(records: list[dict]) -> dict:
    dice_vals = [r["dice"] for r in records]
    return {
        "n_samples": len(records),
        "dice_mean": float(np.mean(dice_vals)) if dice_vals else 0.0,
        "dice_std": float(np.std(dice_vals)) if dice_vals else 0.0,
        "hit_rate": float(np.mean([r["hit"] for r in records])) if records else 0.0,
    }


def _write_summary(output_path: Path, checkpoint: str, hit_threshold: float,
                    records: list[dict], skipped: list[dict]) -> dict:
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)

    summary = {
        "checkpoint": checkpoint,
        "hit_threshold": hit_threshold,
        "overall": summarize(records),
        "by_category": {cat: summarize(recs) for cat, recs in sorted(by_category.items())},
        "n_skipped": len(skipped),
        "skipped": skipped,
        "per_sample": records,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()

    with open(args.manifest) as f:
        samples = json.load(f)
    kept = [s for s in samples if s.get("sentence")]
    n_dropped = len(samples) - len(kept)
    if n_dropped:
        logger.warning(f"Dropping {n_dropped}/{len(samples)} sample(s) with no sentence: "
                        f"{[s['mask'] for s in samples if not s.get('sentence')]}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rcd_dir = Path(args.rcd_dir) if args.rcd_dir else output_path.parent / "sat_raw"
    jsonl_path = output_path.parent / "sat_onc_queries.jsonl"  # kept outside rcd_dir -- inference()
                                                                # itself shutil.copy()s the jsonl into
                                                                # rcd_dir, which would raise
                                                                # SameFileError if we wrote it there too
    entries = build_query_jsonl(kept, args.image_dir, jsonl_path)

    _patch_nib_save_for_long_labels()
    run_sat_inference(args, jsonl_path, rcd_dir)

    if args.visualize:
        viz_dir = Path(args.viz_output_dir) if args.viz_output_dir else output_path.parent / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing visualizations -> {viz_dir}")

    records: list[dict] = []
    skipped: list[dict] = []

    for entry in entries:
        accession = entry["accession"]
        sample_id = sat_sample_id(entry["image_rel"])
        image_path = f"{args.image_dir}/{entry['image_rel']}"

        for sentence, manifest_group in entry["sentence_groups"].items():
            pred_path = sat_output_path(rcd_dir, accession, sample_id, sentence)
            if not pred_path.exists():
                for s in manifest_group:
                    skipped.append({"image": entry["image_rel"], "mask": s["mask"],
                                     "reason": f"missing SAT prediction {pred_path}"})
                continue
            try:
                pred = (nib.load(str(pred_path)).get_fdata() > 0.5).astype(np.float32)
            except Exception as e:
                for s in manifest_group:
                    skipped.append({"image": entry["image_rel"], "mask": s["mask"],
                                     "reason": f"failed to load SAT prediction: {e}"})
                continue

            for s in manifest_group:
                mask_path = f"{args.mask_dir}/{s['mask']}"
                try:
                    image_np, gt = load_gt_mask_on_sat_grid(image_path, mask_path)
                    if gt.shape != pred.shape:
                        raise ValueError(f"SAT prediction/GT shape mismatch: pred={pred.shape} gt={gt.shape}")

                    dice = dice_score_np(pred, gt)
                    cat = s.get("finding") or "Unknown"
                    records.append({
                        "id": s["mask"], "category": cat, "dice": dice,
                        "hit": bool(dice >= args.hit_threshold),
                    })

                    if args.visualize:
                        cat_dir = viz_dir / cat
                        cat_dir.mkdir(parents=True, exist_ok=True)
                        _save_figure(image_np, gt, pred, sentence, dice,
                                     cat_dir / f"{_safe_filename(s['mask'])}.png")
                except Exception as e:
                    logger.warning(f"SKIP mask {s['mask']}: {e}")
                    skipped.append({"image": entry["image_rel"], "mask": s["mask"], "reason": str(e)})

        if len(records) % 100 < len(entry["sentence_groups"]):
            _write_summary(output_path, str(args.checkpoint), args.hit_threshold, records, skipped)

    summary = _write_summary(output_path, str(args.checkpoint), args.hit_threshold, records, skipped)

    overall = summary["overall"]
    logger.info(
        f"Results -> {output_path}\n"
        f"  Overall ({overall['n_samples']} samples, {len(skipped)} skipped):\n"
        f"    Dice: {overall['dice_mean']:.4f} +/- {overall['dice_std']:.4f}   "
        f"Hit: {overall['hit_rate']:.4f}"
    )
    logger.info("  By finding:")
    for cat, s in summary["by_category"].items():
        logger.info(
            f"    {cat:<15} n={s['n_samples']:<4} "
            f"Dice={s['dice_mean']:.4f}+/-{s['dice_std']:.4f}  Hit={s['hit_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
