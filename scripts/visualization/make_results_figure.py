#!/usr/bin/env python3
"""
Journal-figure generator: model predictions vs. GT on N ED cases + N
oncology cases, side by side, white background, colorblind-safe contours.

Two modes
---------
--rank (no figure):
    Run inference on the *entire* ED official test set
    (official_splits/ed_official_test_data.json) and the *entire* oncology
    test set (OncologyTestSetWorksheet.xlsx), print + save a
    Dice-sorted table per cohort (id, dice, sentence -- plus the QC columns
    for oncology: Segmentation Quality Score and the defect Y/N flags), then
    exit. Use this to hand-pick which cases to feature -- nothing is drawn.

(default, figure mode):
    Build the figure from explicit --ed-cases/--onc-cases: comma-separated
    lists of manifest "mask" ids (the "id" column --rank's output prints),
    normally N of each. Runs inference only on those, then saves a
    white-background grid: one row per case (ED cases first, then
    oncology), 3 columns (axial/coronal/sagittal at the GT mask centroid)
    plus a 4th text column (case id, Dice, wrapped sentence). GT contour is
    blue, prediction contour is vermillion (Okabe-Ito palette -- readable in
    grayscale print and colorblind-safe).

Data sources
------------
ED cases: official_splits/ed_official_test_data.json, images under
--image-dir (default nifti_resampled), masks under --ed-mask-dir (default
ED_TEST_SET_resampled) -- same roots evaluate_ed_official_test.py uses.

Oncology cases: OncologyTestSetWorksheet.xlsx (Sheet1, real headers on
the 2nd row -- row 1 is merged group headers). Each row's mask path is
reconstructed as "{Accession}/mask_{Series}_{Slice}_{AnnoIdx}.nii.gz" (the
convention used throughout this repo, e.g. update_box_annotations.py's
mask_rel_path), under --onc-mask-dir (default labels_resampled -- oncology
has no separate curated *_resampled mask dir the way the ED test set does).
Each row's image path isn't in the worksheet, so it's looked up from
official_splits/all_test_data.json by that mask path, falling back to a
plain series-file search under --image-dir/{accession}/ if not found there.
Rows missing Accession/Series/Slice/AnnoIdx/Sentence are skipped.

Requires pandas + openpyxl to read the worksheet (pip install pandas
openpyxl if missing).

Usage
-----
    # 1. rank all candidates by Dice
    python make_results_figure.py --config configs/default.yaml --checkpoint <ckpt> --rank

    # 2. build the figure from N hand-picked ids per cohort
    python make_results_figure.py --config configs/default.yaml --checkpoint <ckpt> \\
        --ed-cases "HEMATOMA/EDDetect.../Struct_....nii.gz,..." \\
        --onc-cases "OncDetect.../mask_2_50_0.nii.gz,..." \\
        --output journal_figure.png
"""
import argparse
import json
import logging
import sys
import tempfile
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import build_dataloader
from models.grounder import Grounder
from train import apply_overrides
from training.trainer import _trim_padding
from utils.metrics import dice_score

DEFAULT_ED_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ed_official_test_data.json"
DEFAULT_ALL_TEST_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "all_test_data.json"
DEFAULT_ONC_WORKSHEET = Path(__file__).resolve().parents[2] / "reference_data" / "OncologyTestSetWorksheet.xlsx"

GT_COLOR = "#0072B2"    # Okabe-Ito blue
PRED_COLOR = "#D55E00"  # Okabe-Ito vermillion
HU_DISPLAY_WINDOW = (-150, 250)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def _mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask > 0.5)
    if len(coords) == 0:
        return tuple(s // 2 for s in mask.shape)
    d, h, w = coords.mean(axis=0).astype(int)
    return int(d), int(h), int(w)


def _trim_volume(vol: np.ndarray, pad_amounts: torch.Tensor) -> np.ndarray:
    pad_D, pad_H, pad_W = (int(p) for p in pad_amounts)
    D = vol.shape[0] - pad_D if pad_D > 0 else vol.shape[0]
    H = vol.shape[1] - pad_H if pad_H > 0 else vol.shape[1]
    W = vol.shape[2] - pad_W if pad_W > 0 else vol.shape[2]
    return vol[:D, :H, :W]


def _normalize_col(s: str) -> str:
    return " ".join(str(s).split()).strip().lower()


def _find_series_file(accession_dir: Path, series: int) -> str | None:
    if not accession_dir.is_dir():
        return None
    for f in sorted(accession_dir.glob("*.nii.gz")):
        prefix = f.name.split("_")[0]
        if prefix.isdigit() and int(prefix) == series:
            return f.name
    return None


def _load_onc_candidates(worksheet_path: Path, all_test_manifest: Path, image_dir: Path) -> list[dict]:
    """Returns [{"image", "mask", "sentence", "quality_score", "defects": [...]}]."""
    df = pd.read_excel(worksheet_path, sheet_name="Sheet1", header=1)
    cols = {_normalize_col(c): c for c in df.columns}

    def col(*keywords: str) -> str:
        # Exact match first: a substring-only search on a single keyword like
        # "sentence" would also match "Sentence Mismatch (Y/N)" (which comes
        # first in column order and is blank for most rows), silently grabbing
        # the wrong column instead of the real "Sentence" text column.
        if len(keywords) == 1 and keywords[0] in cols:
            return cols[keywords[0]]
        for norm, orig in cols.items():
            if all(k in norm for k in keywords):
                return orig
        raise KeyError(f"No column matching {keywords} in {list(df.columns)}")

    accession_col = col("accession")
    score_col = col("segmentation", "quality", "score")
    series_col = col("series")
    slice_col = col("slice")
    annoidx_col = col("annoidx")
    sentence_col = col("sentence")
    defect_cols = {
        "sentence_mismatch": col("sentence", "mismatch"),
        "missing_anatomical_info": col("missing", "anatomical"),
        "under_captured": col("captured"),
        "over_captured": col("too large"),
        "extra_findings": col("extra", "findings"),
    }

    with open(all_test_manifest) as f:
        all_test = json.load(f)
    image_by_mask = {s["mask"]: s["image"] for s in all_test}

    candidates = []
    n_skipped = 0
    for _, row in df.iterrows():
        accession = row.get(accession_col)
        series, slc, anno_idx, sentence = row.get(series_col), row.get(slice_col), row.get(annoidx_col), row.get(sentence_col)
        if pd.isna(accession) or pd.isna(series) or pd.isna(slc) or pd.isna(anno_idx) or pd.isna(sentence):
            n_skipped += 1
            continue

        mask_rel = f"{accession}/mask_{int(series)}_{int(slc)}_{int(anno_idx)}.nii.gz"
        image_rel = image_by_mask.get(mask_rel)
        if image_rel is None:
            found = _find_series_file(image_dir / str(accession), int(series))
            if found is None:
                logger.warning(f"Skipping {mask_rel}: no image found for series {int(series)} under {image_dir / str(accession)}")
                n_skipped += 1
                continue
            image_rel = f"{accession}/{found}"

        raw_score = row.get(score_col)
        candidates.append({
            "image": image_rel,
            "mask": mask_rel,
            "sentence": str(sentence),
            "quality_score": None if pd.isna(raw_score) else float(raw_score),
            "defects": [name for name, c in defect_cols.items() if str(row.get(c)).strip().upper() == "Y"],
        })

    logger.info(f"Oncology worksheet: {len(candidates)} usable case(s), {n_skipped} skipped (missing fields or image)")
    return candidates


def _load_ed_candidates(manifest_path: Path) -> list[dict]:
    with open(manifest_path) as f:
        samples = json.load(f)
    kept = [s for s in samples if s.get("sentence")]
    if len(kept) < len(samples):
        logger.warning(f"Dropping {len(samples) - len(kept)} ED sample(s) with no sentence")
    return kept


def _build_model(cfg: dict, checkpoint: str, device: torch.device, use_cached_text: bool) -> Grounder:
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
        fusion_type=cfg["model"].get("fusion_type", "cross_attention"),
        voxtell_guidance_dim=cfg["model"].get("voxtell", {}).get("guidance_dim", 32),
        voxtell_prompt_decoder_dim=cfg["model"].get("voxtell", {}).get("prompt_decoder_dim", 256),
        voxtell_prompt_decoder_layers=cfg["model"].get("voxtell", {}).get("prompt_decoder_layers", 6),
        voxtell_prompt_decoder_heads=cfg["model"].get("voxtell", {}).get("prompt_decoder_heads", 8),
        load_text_backbone=not use_cached_text,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    return model


def _run_inference(
    loader, model: Grounder, device: torch.device, use_cached_text: bool, threshold: float, keep_volumes: bool
) -> list[dict]:
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if use_cached_text:
                    logits = model(image, text_feats=batch["text_feats"].to(device),
                                   text_padding_mask=batch["text_padding_mask"].to(device))
                else:
                    logits = model(image, batch["input_ids"].to(device), batch["attention_mask"].to(device))

            logits, trimmed_mask = _trim_padding(logits, mask, batch["pad_amounts"])
            dices = dice_score(logits, trimmed_mask, threshold=threshold, from_logits=True)

            for i, sample_id in enumerate(batch["id"]):
                rec = {"id": sample_id, "dice": dices[i].item()}
                if keep_volumes:
                    rec["image"] = _trim_volume(image[i, 0].cpu().float().numpy(), batch["pad_amounts"][i])
                    rec["gt"] = trimmed_mask[i, 0].cpu().float().numpy()
                    rec["pred"] = (torch.sigmoid(logits[i, 0]).cpu().float().numpy() > threshold).astype(np.float32)
                results.append(rec)
    return results


def _save_journal_figure(cases: list[dict], out_path: Path, dpi: int) -> None:
    """cases: [{"id", "sentence", "dice", "image", "gt", "pred"}, ...], ED first then oncology."""
    lo, hi = HU_DISPLAY_WINDOW
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.6 * n), facecolor="white",
                              gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Axial", "Coronal", "Sagittal"]
    for row_idx, case in enumerate(cases):
        d, h, w = _mask_centroid(case["gt"])
        views = [
            (case["image"][d, :, :], case["gt"][d, :, :], case["pred"][d, :, :]),
            (case["image"][:, h, :], case["gt"][:, h, :], case["pred"][:, h, :]),
            (case["image"][:, :, w], case["gt"][:, :, w], case["pred"][:, :, w]),
        ]
        for col_idx, (img_sl, gt_sl, pred_sl) in enumerate(views):
            ax = axes[row_idx, col_idx]
            img_sl = np.rot90(np.clip(img_sl, lo, hi), 2)
            gt_sl = np.rot90(gt_sl, 2)
            pred_sl = np.rot90(pred_sl, 2)
            ax.imshow(img_sl, cmap="gray", aspect="equal", origin="upper")
            if gt_sl.any():
                ax.contour(gt_sl, levels=[0.5], colors=[GT_COLOR], linewidths=1.8)
            if pred_sl.any():
                ax.contour(pred_sl, levels=[0.5], colors=[PRED_COLOR], linewidths=1.8)
            ax.set_facecolor("white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#888888")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12)

        text_ax = axes[row_idx, 3]
        text_ax.axis("off")
        label = f"{case['id']}\nDice = {case['dice']:.3f}\n\n" + textwrap.fill(case["sentence"], width=42)
        text_ax.text(0.0, 0.5, label, fontsize=9, va="center", ha="left", transform=text_ax.transAxes, wrap=True)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color=GT_COLOR, linewidth=1.8, label="Ground truth"),
        Line2D([0], [0], color=PRED_COLOR, linewidth=1.8, label="Prediction"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0 + 0.4 / n), frameon=False, fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", default="/path/to/data/inhouse_abdominal_ct/nifti_resampled")
    parser.add_argument("--ed-mask-dir", default="/path/to/data/inhouse_abdominal_ct/ED_TEST_SET_resampled")
    parser.add_argument("--onc-mask-dir", default="/path/to/data/inhouse_abdominal_ct/labels_resampled")
    parser.add_argument("--ed-manifest", type=Path, default=DEFAULT_ED_MANIFEST)
    parser.add_argument("--onc-worksheet", type=Path, default=DEFAULT_ONC_WORKSHEET)
    parser.add_argument("--all-test-manifest", type=Path, default=DEFAULT_ALL_TEST_MANIFEST)
    parser.add_argument("--n", type=int, default=4, help="Expected cases per cohort in figure mode (validation only)")
    parser.add_argument("--rank", action="store_true", help="Rank all candidates by Dice and exit, instead of building a figure")
    parser.add_argument("--rank-output", default="results_ranking.json")
    parser.add_argument("--ed-cases", default=None, help="Comma-separated mask ids to feature (figure mode)")
    parser.add_argument("--onc-cases", default=None, help="Comma-separated mask ids to feature (figure mode)")
    parser.add_argument("--output", default="outputs/figures/journal_figure.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--override", nargs="*", default=[], help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = cfg["inference"].get("threshold", 0.5)
    use_cached_text = bool(cfg["data"].get("embedding_cache"))

    ed_candidates = _load_ed_candidates(args.ed_manifest)
    onc_candidates = _load_onc_candidates(args.onc_worksheet, args.all_test_manifest, Path(args.image_dir))

    if not args.rank:
        if not args.ed_cases or not args.onc_cases:
            sys.exit("Figure mode requires --ed-cases and --onc-cases (comma-separated mask ids). "
                     "Run with --rank first to get a Dice-sorted list of ids to pick from.")
        ed_ids = [s.strip() for s in args.ed_cases.split(",") if s.strip()]
        onc_ids = [s.strip() for s in args.onc_cases.split(",") if s.strip()]
        ed_by_id = {c["mask"]: c for c in ed_candidates}
        onc_by_id = {c["mask"]: c for c in onc_candidates}
        missing = [i for i in ed_ids if i not in ed_by_id] + [i for i in onc_ids if i not in onc_by_id]
        if missing:
            sys.exit(f"These case ids weren't found among the candidates: {missing}")
        ed_candidates = [ed_by_id[i] for i in ed_ids]
        onc_candidates = [onc_by_id[i] for i in onc_ids]
        if len(ed_candidates) != args.n or len(onc_candidates) != args.n:
            logger.warning(f"--n={args.n} but got {len(ed_candidates)} ED case(s) and {len(onc_candidates)} oncology case(s)")

    model = _build_model(cfg, args.checkpoint, device, use_cached_text)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"image": c["image"], "mask": c["mask"], "sentence": c["sentence"]} for c in ed_candidates], f)
        ed_manifest_tmp = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([{"image": c["image"], "mask": c["mask"], "sentence": c["sentence"]} for c in onc_candidates], f)
        onc_manifest_tmp = f.name

    entries = [
        {"manifest": ed_manifest_tmp, "image_dir": args.image_dir, "mask_dir": args.ed_mask_dir},
        {"manifest": onc_manifest_tmp, "image_dir": args.image_dir, "mask_dir": args.onc_mask_dir},
    ]
    cfg_infer = {**cfg, "training": {**cfg["training"], "batch_size": 1 if not args.rank else cfg["training"]["batch_size"]}}
    loader = build_dataloader(entries, cfg_infer, split="test", num_workers=0)

    keep_volumes = not args.rank
    results = _run_inference(loader, model, device, use_cached_text, threshold, keep_volumes)

    ed_ids_set = {c["mask"] for c in ed_candidates}
    sentence_by_id = {c["mask"]: c["sentence"] for c in ed_candidates + onc_candidates}
    qc_by_id = {c["mask"]: c for c in onc_candidates}

    if args.rank:
        ed_ranked = sorted([r for r in results if r["id"] in ed_ids_set], key=lambda r: r["dice"], reverse=True)
        onc_ranked = sorted([r for r in results if r["id"] not in ed_ids_set], key=lambda r: r["dice"], reverse=True)

        def _row(r: dict) -> dict:
            out = {"id": r["id"], "dice": round(r["dice"], 4), "sentence": sentence_by_id.get(r["id"], "")}
            if r["id"] in qc_by_id:
                out["quality_score"] = qc_by_id[r["id"]]["quality_score"]
                out["defects"] = qc_by_id[r["id"]]["defects"]
            return out

        ranking = {"ed": [_row(r) for r in ed_ranked], "oncology": [_row(r) for r in onc_ranked]}
        with open(args.rank_output, "w") as f:
            json.dump(ranking, f, indent=2)

        for cohort_name, ranked in [("ED", ed_ranked), ("Oncology", onc_ranked)]:
            logger.info(f"\n{cohort_name} ({len(ranked)} cases), best to worst:")
            for r in ranked:
                extra = ""
                if r["id"] in qc_by_id:
                    qc = qc_by_id[r["id"]]
                    extra = f"  qc_score={qc['quality_score']}  defects={qc['defects'] or 'none'}"
                logger.info(f"  dice={r['dice']:.3f}  {r['id']}{extra}  -- {sentence_by_id.get(r['id'], '')[:80]}")
        logger.info(f"\nFull ranking written to {args.rank_output}")
        return

    ed_results = [r for r in results if r["id"] in ed_ids_set]
    onc_results = [r for r in results if r["id"] not in ed_ids_set]
    # Preserve the order the user specified in --ed-cases/--onc-cases, not inference order
    ed_by_result_id = {r["id"]: r for r in ed_results}
    onc_by_result_id = {r["id"]: r for r in onc_results}
    missing_results = [i for i in ed_ids if i not in ed_by_result_id] + [i for i in onc_ids if i not in onc_by_result_id]
    if missing_results:
        sys.exit(f"These case(s) didn't come back from inference -- GrounderDataset likely dropped them because "
                  f"the image/mask file wasn't found under --image-dir/--ed-mask-dir/--onc-mask-dir "
                  f"(check the warnings above): {missing_results}")
    ordered = [ed_by_result_id[i] for i in ed_ids] + [onc_by_result_id[i] for i in onc_ids]
    for r in ordered:
        r["sentence"] = sentence_by_id.get(r["id"], r["id"])

    _save_journal_figure(ordered, Path(args.output), args.dpi)
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
