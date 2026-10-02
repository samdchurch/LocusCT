#!/usr/bin/env python3
"""
Convert an existing embeddings.pt cache (torch dict format) to the numpy
memmap directory format used by GrounderDataset.

All DDP ranks mmap the same output files, so the OS page cache holds one
shared copy in RAM regardless of how many ranks are running — eliminating
the N_ranks x cache_size RAM duplication from torch.load.

Note: requires enough RAM to load the full .pt file once during conversion.

Usage
-----
    python convert_embedding_cache.py \
        --input /path/to/embeddings.pt \
        --output /path/to/embeddings_mmap
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  required=True, help="Path to existing embeddings.pt")
    parser.add_argument("--output", required=True, help="Output directory for memmap cache")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", args.input)
    cache: dict = torch.load(args.input, map_location="cpu")

    sample_ids = list(cache.keys())
    N = len(sample_ids)
    first = cache[sample_ids[0]]
    L, D = first["text_feats"].shape
    log.info("N=%d  L=%d  D=%d", N, L, D)

    feats_mm = np.lib.format.open_memmap(
        str(out_dir / "text_feats.npy"), mode="w+", dtype=np.float32, shape=(N, L, D)
    )
    masks_mm = np.lib.format.open_memmap(
        str(out_dir / "text_padding_mask.npy"), mode="w+", dtype=bool, shape=(N, L)
    )

    index: dict[str, int] = {}
    for i, sid in enumerate(sample_ids):
        feats_mm[i] = cache[sid]["text_feats"].numpy()
        masks_mm[i] = cache[sid]["text_padding_mask"].numpy()
        index[sid] = i
        del cache[sid]  # free RAM progressively
        if (i + 1) % 10_000 == 0:
            log.info("  %d / %d", i + 1, N)

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)

    log.info("Done. Written to %s", out_dir)
    log.info("  text_feats.npy:        %.1f GB", (N * L * D * 4) / 2**30)
    log.info("  text_padding_mask.npy: %.1f MB", (N * L) / 2**20)


if __name__ == "__main__":
    main()
