"""
Precompute and cache raw text encoder hidden states for all manifest samples.

Run this once before training with embedding_cache enabled:
    python precompute_embeddings.py --config configs/default.yaml --output /path/to/embeddings_mmap
    python precompute_embeddings.py --config configs/default.yaml --text-field in-context --output /path/to/incontext_embeddings_mmap

The output directory contains:
    text_feats.npy          (N, L, hidden_size) float32 — memory-mappable
    text_padding_mask.npy   (N, L)              bool    — memory-mappable
    index.json              {sample_id: row_index}

All DDP ranks mmap the same files, so the OS page cache holds one shared
copy in RAM regardless of how many ranks are running.

These are the frozen backbone's raw, unprojected hidden states -- there is no
shared projection layer to keep in sync across cache-building runs. Each
cross-attention module (one per UNet decoder/encoder stage) owns its own
k_proj/v_proj mapping from this raw hidden size, and those live in and train
with the UNet, so they're always part of the checkpoint regardless of
embedding_cache. Building a second cache (e.g. for a held-out test set) is
therefore safe to do independently -- no projection to keep consistent.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from models.text_encoder import TextEncoder
from train import apply_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--text-field", default="sentence",
                         help="Manifest field to embed -- 'sentence' (the bare finding text, default) or "
                              "'in-context' (the surrounding report section with the finding wrapped in "
                              "<REF></REF> tags). Produces a separate cache; run once per field into "
                              "different --output directories to have both available.")
    parser.add_argument("--fallback-field", default=None,
                         help="If a sample is missing --text-field (e.g. ~18%% of samples have no "
                              "'in-context' text), use this field instead of dropping the sample entirely "
                              "-- e.g. --text-field in-context --fallback-field sentence. Unset by default "
                              "(samples missing --text-field are dropped, with a warning).")
    parser.add_argument("--output", required=True, help="Output directory for memmap cache")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    encoder = TextEncoder(
        model_name=cfg["model"]["text_encoder_name"],
        freeze_backbone=True,
    ).to(device)
    encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder_name"])
    max_len = cfg["data"]["max_text_len"]
    hidden_size = encoder.transformer.config.hidden_size

    # Collect unique samples from all configured manifests
    manifest_paths: list[str] = []
    for key in ("train_manifest", "val_manifest", "test_manifest"):
        v = cfg["data"].get(key)
        if v is None:
            continue
        manifest_paths.extend([v] if isinstance(v, str) else v)

    samples: dict[str, str] = {}  # mask path (id) -> text (deduplicated)
    n_fallback = 0
    n_missing = 0
    for path in manifest_paths:
        with open(path) as f:
            data = json.load(f)
        for s in data:
            text = s.get(args.text_field)
            if not text and args.fallback_field:
                text = s.get(args.fallback_field)
                if text:
                    n_fallback += 1
            if not text:
                n_missing += 1
                continue
            samples[s["mask"]] = text
    if n_fallback:
        logger.warning(f"Used '{args.fallback_field}' fallback for {n_fallback} sample(s) missing '{args.text_field}'")
    if n_missing:
        reason = f"'{args.text_field}'" + (f" or fallback '{args.fallback_field}'" if args.fallback_field else "")
        logger.warning(f"Skipping {n_missing} sample(s) with no {reason} field")

    N = len(samples)
    logger.info(f"Computing embeddings for {N} samples across {len(manifest_paths)} manifests "
                f"(hidden_size={hidden_size})")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate memmap arrays on disk — written incrementally so RAM usage
    # stays constant regardless of N.
    feats_mm = np.lib.format.open_memmap(
        str(out_dir / "text_feats.npy"), mode="w+", dtype=np.float32, shape=(N, max_len, hidden_size)
    )
    masks_mm = np.lib.format.open_memmap(
        str(out_dir / "text_padding_mask.npy"), mode="w+", dtype=bool, shape=(N, max_len)
    )

    index: dict[str, int] = {}

    with torch.no_grad():
        for i, (sample_id, expression) in enumerate(tqdm(samples.items(), desc="Encoding")):
            encoding = tokenizer(
                expression,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            text_feats, text_padding_mask = encoder(input_ids, attention_mask)
            feats_mm[i] = text_feats.squeeze(0).float().cpu().numpy()
            masks_mm[i] = text_padding_mask.squeeze(0).cpu().numpy()
            index[sample_id] = i

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)

    logger.info(f"Saved {N} embeddings to {out_dir}")


if __name__ == "__main__":
    main()
