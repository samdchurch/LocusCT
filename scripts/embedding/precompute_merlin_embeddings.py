#!/usr/bin/env python3
"""
Precompute and cache text encoder hidden states for merlin_sentences.json's
atomic findings -- same cache format precompute_embeddings.py builds for
training manifests (text_feats.npy (N, L, hidden_size) float32 memmap,
text_padding_mask.npy (N, L) bool memmap, index.json {sample_id: row_index}),
but keyed by predict_merlin.sample_key(study_id, finding_idx) rather than a
manifest's "mask" path, since Merlin samples have no mask file.

Needed because checkpoints trained with data.embedding_cache set (the
default in configs/default.yaml and configs/rexgroundingct_finetune.yaml)
never load a live TextEncoder, so predict_merlin.py can't tokenize Merlin's
previously-unseen sentences on the fly with those checkpoints. Point
predict_merlin.py at the resulting directory via --embedding-cache to use
it instead of live encoding (much faster too, since the 8B-parameter frozen
backbone then never runs at inference time) -- see that script's module
docstring for the load_text_backbone / strict-loading tradeoff either way.

Reuses predict_merlin.load_merlin_samples() so the exact same (study,
finding, status, normal-finding) filtering applies here as at inference
time -- rebuild this cache if --sentences/--image-dir/--status/
--include-normal ever change, or predict_merlin.py will refuse to run
against a stale one (it checks that every requested sample's key is
present in the cache's index).

Usage
-----
    python precompute_merlin_embeddings.py --config configs/default.yaml --output /path/to/merlin_embeddings_mmap
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "evaluation"))  # for predict_merlin

from models.text_encoder import TextEncoder
from predict_merlin import DEFAULT_IMAGE_DIR, DEFAULT_SENTENCES, load_merlin_samples, sample_key
from train import apply_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                         help="Only studies with a volume under here are included -- must match predict_merlin.py's --image-dir")
    parser.add_argument("--status", nargs="+", default=["present"],
                         help="Only cache atomic_findings with one of these status values (default: present) -- "
                              "must match predict_merlin.py's --status")
    parser.add_argument("--include-normal", action="store_true",
                         help="Include normal/unremarkable/patent findings (excluded by default) -- "
                              "must match predict_merlin.py's --include-normal")
    parser.add_argument("--output", required=True, help="Output directory for the memmap cache")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    encoder = TextEncoder(model_name=cfg["model"]["text_encoder_name"], freeze_backbone=True).to(device)
    encoder.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["text_encoder_name"])
    max_len = cfg["data"]["max_text_len"]
    hidden_size = encoder.transformer.config.hidden_size

    samples = load_merlin_samples(args.sentences, args.image_dir, args.status, skip_normal=not args.include_normal)
    if not samples:
        logger.error("No samples to cache -- check --sentences/--image-dir/--status")
        sys.exit(1)

    N = len(samples)
    logger.info(f"Computing embeddings for {N} (study, finding) sample(s) (hidden_size={hidden_size})")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate memmap arrays on disk -- written incrementally so RAM usage stays
    # constant regardless of N (same approach as precompute_embeddings.py).
    feats_mm = np.lib.format.open_memmap(
        str(out_dir / "text_feats.npy"), mode="w+", dtype=np.float32, shape=(N, max_len, hidden_size)
    )
    masks_mm = np.lib.format.open_memmap(
        str(out_dir / "text_padding_mask.npy"), mode="w+", dtype=bool, shape=(N, max_len)
    )

    index: dict[str, int] = {}

    with torch.no_grad():
        for i, sample in enumerate(tqdm(samples, desc="Encoding")):
            encoding = tokenizer(
                sample["sentence"], max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            text_feats, text_padding_mask = encoder(input_ids, attention_mask)
            feats_mm[i] = text_feats.squeeze(0).float().cpu().numpy()
            masks_mm[i] = text_padding_mask.squeeze(0).cpu().numpy()
            index[sample_key(sample["study_id"], sample["finding_idx"])] = i

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)

    logger.info(f"Saved {N} embeddings to {out_dir}")


if __name__ == "__main__":
    main()
