#!/usr/bin/env python3
"""
Runs a Grounder checkpoint over the official (blind) ReXGroundingCT test set
(official_splits/ReXGroundingCT_test.json -- 582 findings across 300 unique
volumes, every entry has "mask": null) and saves one native-resolution
prediction NIfTI per volume to <run_dir>/official_predictions/, where
<run_dir> is the checkpoint's own run directory (parent of its checkpoints/
folder) -- for submission to the ReXGroundingCT leaderboard.

Output filenames and per-volume finding count/shape are validated against
/path/to/data/public_datasets/ReXGroundingCT/MICCAI_challenge_dataset.json
(--challenge-json), the authoritative source for what a submission entry
should look like: each output file is named exactly "<name>" from that
JSON's "test" split (e.g. "train_13195_a_1.nii.gz" -- the trailing digit is
part of the volume's own name, NOT a finding index, see
_TestOnlyGrounderDataset), and must stack exactly len(findings) masks along
axis 0 at the volume's declared native (H, W, D) "shape". A volume that
doesn't match is skipped (see skipped.json) rather than silently written
wrong.

Not a --test mode of evaluate_rexgroundingct_val.py: that script's whole
second half (GT stacking + rexrank_eval.py scoring) requires ground truth
the test set doesn't have, so this is a separate, inference-only script.

GrounderDataset can't load these entries as-is -- it requires a real mask
file (_filter_missing checks mask_path.exists(), __getitem__ uses
sample["mask"] as both the file to load and the sample "id"). Rather than
special-case data/dataset.py's shared class (used by train.py and every
other eval script), _TestOnlyGrounderDataset below overrides just the
mask-handling: skip the existence check, synthesize a same-shape zero array
in place of a real mask (its value is irrelevant -- never used for anything
but must survive the shared pad/resize pipeline), and use sample["image"]
as "id" instead. It also assigns each sample its true (volume_name,
finding_idx) -- see the class docstring for why that can't just be parsed
from the image filename the way mask filenames elsewhere in this project
can.

Native-resolution resampling reuses evaluate_ed_official_test.py's
_canonical_mask_affine() -- the model's prediction lives on the resampled
grid, whose on-disk affine already correctly maps to world space; that
affine is carried through GrounderDataset's canonicalization so the
prediction can be resampled (nibabel resample_from_to, nearest-neighbor)
onto the native grid without hand-inverting any reorientation.

Usage
-----
    python predict_rexgroundingct_test.py --config configs/default.yaml --checkpoint runs/h200/my_run/checkpoints/best.pt
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from nibabel.processing import resample_from_to
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import GrounderDataset, load_nifti_canonical
from evaluate_ed_official_test import _canonical_mask_affine
from models.grounder import Grounder
from train import apply_overrides
from training.trainer import _trim_padding

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "ReXGroundingCT_test.json"
DEFAULT_DATA_ROOT = Path("/path/to/data/public_datasets")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True,
                         help="Output goes to this checkpoint's own run dir (parent of checkpoints/), "
                              "not a chosen --output-dir")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                         help="Base dir the manifest's relative image paths resolve against")
    parser.add_argument("--challenge-json", type=Path, default=None,
                         help="MICCAI_challenge_dataset.json, authoritative source for output filenames "
                              "and per-volume finding count/native shape (default: <data-root>/ReXGroundingCT/"
                              "MICCAI_challenge_dataset.json)")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model",
    )
    return parser.parse_args()


def native_image_path(data_root: Path, resampled_image_rel: str) -> Path:
    """'ReXGroundingCT/resampled/images/...' -> 'ReXGroundingCT/original/images/...'"""
    native_rel = resampled_image_rel.replace("ReXGroundingCT/resampled/", "ReXGroundingCT/original/", 1)
    return data_root / native_rel


class _TestOnlyGrounderDataset(GrounderDataset):
    """GrounderDataset variant for manifests with no real mask file (mask: null) --
    see module docstring for why this overrides rather than touching the shared class.

    Also assigns each sample its true (volume_name, finding_idx): every finding for a
    given volume shares the exact same "image" path (one physical CT per volume, not
    per finding), so unlike resample_rexgroundingct.py's per-finding mask filenames,
    there's no finding index encoded in the filename to parse back out -- the trailing
    "_<digit>" there is part of the volume's own name (e.g. "train_13195_a_1.nii.gz"'s
    "_1"), not a finding index. The true finding index is just this sample's position
    among same-image samples in manifest order, verified (see conversation/commit that
    introduced this) to exactly match MICCAI_challenge_dataset.json's "findings" dict
    key order -- same name, same count, same sentence order -- for every one of the
    300 official test volumes.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        counters: dict[str, int] = defaultdict(int)
        for sample in self.samples:
            name = Path(sample["image"]).name  # matches MICCAI_challenge_dataset.json's "name" field
            sample["volume_name"] = name
            sample["finding_idx"] = counters[name]
            counters[name] += 1

    def _filter_missing(self, samples: list[dict]) -> list[dict]:
        kept = []
        for sample in samples:
            image_path = self.image_dir / sample["image"] if self.image_dir else Path(sample["image"])
            if not image_path.exists():
                logger.warning(f"Skipping missing image: {sample['image']}")
                continue
            kept.append(sample)
        if len(kept) < len(samples):
            logger.warning(f"Dropped {len(samples) - len(kept)}/{len(samples)} samples with missing images")
        return kept

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.image_dir / sample["image"] if self.image_dir else sample["image"]
        try:
            image_np = load_nifti_canonical(image_path)
        except Exception:
            logger.warning(f"Failed to load image for sample {sample['image']}; skipping to next sample",
                            exc_info=True)
            return self.__getitem__((idx + 1) % len(self.samples))

        mask_np = np.zeros(image_np.shape[-3:], dtype=np.float32)  # unused placeholder, see module docstring
        image_np = self._apply_windows(image_np)

        if self.spatial_mode == "fixed":
            image, mask, pad_amounts = self._pad_to_divisible(image_np, mask_np)
        else:
            image = self._resize_volume(image_np, self.spatial_size, is_mask=False)
            mask = self._resize_volume(mask_np, self.spatial_size, is_mask=True)
            pad_amounts = torch.zeros(3, dtype=torch.long)

        item = {
            "image": image, "mask": mask.float(), "id": sample["image"], "pad_amounts": pad_amounts,
            "volume_name": sample["volume_name"], "finding_idx": sample["finding_idx"],
        }
        # Always live-encode -- any real embedding_cache is keyed by manifest "mask"
        # path (precompute_embeddings.py), which doesn't exist for this test
        # manifest (mask: null everywhere), so it can't be reused here. 582
        # sentences is trivial to tokenize live; see __init__'s embedding_cache="".
        encoding = self.tokenizer(
            sample["sentence"], max_length=self.max_text_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        item["input_ids"] = encoding["input_ids"].squeeze(0)
        item["attention_mask"] = encoding["attention_mask"].squeeze(0)
        return item


def main() -> None:
    args = parse_args()

    challenge_json_path = args.challenge_json or (args.data_root / "ReXGroundingCT" / "MICCAI_challenge_dataset.json")
    with open(challenge_json_path) as f:
        challenge_data = json.load(f)
    challenge_index: dict[str, dict] = {e["name"]: e for e in challenge_data["test"]}
    logger.info(f"Loaded {len(challenge_index)} test volume(s) from {challenge_json_path}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = cfg["inference"].get("threshold", 0.5)
    # Always live-encode text for this script -- see _TestOnlyGrounderDataset.__getitem__'s
    # docstring comment for why any real embedding_cache can't be reused here.

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
        load_text_backbone=True,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # strict=False: checkpoints trained with a cached text-embedding model
    # (load_text_backbone=False) never saved the frozen transformer's weights
    # at all, since that submodule wasn't instantiated -- but this script
    # always loads the real backbone (load_text_backbone=True above, since
    # the test manifest has no embedding cache to reuse), so those "missing"
    # keys are just the pretrained HF weights already loaded via
    # from_pretrained, never touched by training either way.
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    non_text_missing = [k for k in missing if not k.startswith("text_encoder.transformer.")]
    if non_text_missing or unexpected:
        raise RuntimeError(
            f"Unexpected state_dict mismatch beyond the frozen text backbone: "
            f"{len(non_text_missing)} missing (non-text-encoder), {len(unexpected)} unexpected -- "
            f"missing={non_text_missing}  unexpected={unexpected}"
        )
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
                f"({len(missing)} frozen text-encoder key(s) loaded fresh from pretrained instead)")

    dataset = _TestOnlyGrounderDataset(
        manifest_path=str(args.manifest),
        tokenizer_name=cfg["model"]["text_encoder_name"],
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        max_text_len=cfg["data"]["max_text_len"],
        hu_min=cfg["data"]["hu_min"],
        hu_max=cfg["data"]["hu_max"],
        multi_window=cfg["data"].get("multi_window", False),
        spatial_mode=cfg["data"]["spatial_mode"],
        augment=False,
        image_dir=str(args.data_root),
        mask_dir=str(args.data_root),
        embedding_cache="",
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0, pin_memory=True,
    )
    logger.info(f"Test samples: {len(dataset)}")

    pred_by_volume: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    image_rel_by_volume: dict[str, str] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            image = batch["image"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                logits = model(image, input_ids, attn_mask)
            logits, _ = _trim_padding(logits, batch["mask"], batch["pad_amounts"])
            probs = torch.sigmoid(logits).cpu().float().numpy()

            for i, volume_name in enumerate(batch["volume_name"]):
                finding_idx = int(batch["finding_idx"][i])
                pred_np = (probs[i, 0] > threshold).astype(np.uint8)  # (D, H, W), canonical RAS+
                pred_by_volume[volume_name][finding_idx] = pred_np
                image_rel_by_volume.setdefault(volume_name, batch["id"][i])

    run_dir = Path(args.checkpoint).resolve().parent.parent
    out_dir = run_dir / "official_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing official predictions -> {out_dir}")

    skipped: list[dict] = []
    n_written = 0
    for volume_name, finding_preds in tqdm(sorted(pred_by_volume.items()), desc="Resampling to native + saving"):
        try:
            challenge_entry = challenge_index.get(volume_name)
            if challenge_entry is None:
                skipped.append({"volume": volume_name, "reason": "not present in MICCAI_challenge_dataset.json test split"})
                continue

            expected_indices = list(range(len(challenge_entry["findings"])))
            finding_indices = sorted(finding_preds)
            if finding_indices != expected_indices:
                skipped.append({
                    "volume": volume_name,
                    "reason": f"finding indices {finding_indices} != expected {expected_indices} per MICCAI_challenge_dataset.json",
                })
                continue
            pred_4d_model_space = np.stack([finding_preds[i] for i in finding_indices], axis=0)

            resampled_image_rel = image_rel_by_volume[volume_name]
            resampled_img = nib.load(str(args.data_root / resampled_image_rel))
            affine = _canonical_mask_affine(resampled_img)

            native_path = native_image_path(args.data_root, resampled_image_rel)
            if not native_path.exists():
                skipped.append({"volume": volume_name, "reason": "native_image_not_found", "expected_path": str(native_path)})
                continue
            native_img = nib.load(str(native_path))

            native_slices = []
            for finding_slice in pred_4d_model_space:
                mask_img = nib.Nifti1Image(finding_slice, affine)
                resampled_native = resample_from_to(mask_img, native_img, order=0, mode="constant", cval=0)
                native_slices.append(np.asanyarray(resampled_native.dataobj).astype(np.uint8))
            pred_4d_native = np.stack(native_slices, axis=0)

            expected_shape = list(challenge_entry["shape"])
            if list(pred_4d_native.shape[1:]) != expected_shape:
                skipped.append({
                    "volume": volume_name,
                    "reason": f"native shape {list(pred_4d_native.shape[1:])} != MICCAI_challenge_dataset.json shape {expected_shape}",
                })
                continue

            nib.save(nib.Nifti1Image(pred_4d_native, native_img.affine), str(out_dir / volume_name))
            n_written += 1
        except Exception as e:
            logger.warning(f"SKIP {volume_name}: {e}", exc_info=True)
            skipped.append({"volume": volume_name, "reason": str(e)})

    if skipped:
        with open(out_dir / "skipped.json", "w") as f:
            json.dump(skipped, f, indent=2)
        logger.warning(f"{len(skipped)} volume(s) skipped, see skipped.json")

    logger.info(f"Wrote {n_written}/{len(pred_by_volume)} volume(s) to {out_dir}")


if __name__ == "__main__":
    main()
