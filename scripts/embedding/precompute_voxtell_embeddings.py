#!/usr/bin/env python3
"""
Precompute and cache VoxTell's pooled text embeddings for finetune_voxtell.py's
train/val manifests, so fine-tuning can load them from disk via
--embedding-cache instead of running the live Qwen3-Embedding-4B backbone --
which then never needs to be loaded during training at all (frees ~4B
parameters' worth of GPU memory per DDP rank, on top of skipping redundant
re-encoding of the same frozen-backbone output on every epoch).

Calls finetune_voxtell.py's own load_text_backbone()/embed_sentences()
directly (not reimplemented here), so cached values are guaranteed identical
to whatever live encoding would have produced -- same instruction-wrapping/
last-token-pooling, same text encoder.

Unlike precompute_embeddings.py's cache (per-token hidden states, for
Grounder's cross-attention), VoxTell's embed_sentences() pools each sentence
down to a single (1, text_dim) vector before the model ever sees it, so this
cache is a single flat array -- no padding-mask file needed.

Output directory contains:
    text_embeds.npy   (N, text_dim) float32 -- memory-mappable
    index.json        {mask_path: row_index}

Samples are deduplicated by manifest "mask" path across every --manifests
file (same convention precompute_embeddings.py uses), so the default (train
+ val) writes one shared cache covering both splits.

Usage
-----
    python precompute_voxtell_embeddings.py --output /path/to/voxtell_embeddings_mmap
    python precompute_voxtell_embeddings.py --text-field in-context --output /path/to/voxtell_incontext_embeddings_mmap
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from finetune_voxtell import (
    DEFAULT_TEXT_ENCODER,
    DEFAULT_TRAIN_MANIFEST,
    DEFAULT_VAL_MANIFEST,
    embed_sentences,
    load_text_backbone,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", default=[DEFAULT_TRAIN_MANIFEST, DEFAULT_VAL_MANIFEST],
                         help=f"Manifests to collect (mask, sentence) pairs from "
                              f"(default: {[DEFAULT_TRAIN_MANIFEST, DEFAULT_VAL_MANIFEST]})")
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
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--output", required=True, help="Output directory for the memmap cache")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    samples: dict[str, str] = {}  # mask path (id) -> text (deduplicated)
    n_fallback = 0
    n_missing = 0
    for path in args.manifests:
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
    logger.info(f"Computing embeddings for {N} sample(s) across {len(args.manifests)} manifest(s)")

    logger.info(f"Loading frozen text backbone {args.text_encoder}")
    tokenizer, backbone = load_text_backbone(args.text_encoder, device)
    text_dim = backbone.config.hidden_size
    if text_dim != 2560:
        logger.warning(
            f"Text encoder hidden_size={text_dim}, expected 2560 -- build_voxtell_model hardcodes "
            f"text_embedding_dim=2560 to match the pretrained VoxTell checkpoint, so this cache "
            f"likely won't load correctly. Double check --text-encoder is the 4B model."
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocated memmap, written incrementally so RAM usage stays constant
    # regardless of N (same approach as precompute_embeddings.py).
    embeds_mm = np.lib.format.open_memmap(
        str(out_dir / "text_embeds.npy"), mode="w+", dtype=np.float32, shape=(N, text_dim)
    )

    index: dict[str, int] = {}

    for i, (sample_id, sentence) in enumerate(tqdm(samples.items(), desc="Encoding")):
        emb = embed_sentences(tokenizer, backbone, [sentence], device)  # (1, 1, text_dim)
        embeds_mm[i] = emb.squeeze(0).squeeze(0).cpu().numpy()
        index[sample_id] = i

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)

    logger.info(f"Saved {N} embeddings to {out_dir}")


if __name__ == "__main__":
    main()
