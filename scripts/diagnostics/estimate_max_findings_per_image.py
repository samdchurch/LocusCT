#!/usr/bin/env python3
"""
Estimates a safe value for data.max_findings_per_image (see
data.dataset.GrounderImageGroupedDataset), two ways:

1. Data distribution: groups the given manifest by "image" and reports the
   findings-per-image histogram (min/p50/p90/p95/p99/max) -- shows what
   fraction of real images a given cap would actually truncate. Pure JSON
   grouping, no file I/O -- so this is a count of manifest entries, not
   filtered by whether the image/mask files actually exist on disk.

2. GPU memory probe: builds the real Grounder model at the given config
   (same architecture/spatial_size/fusion_type your training run uses) and
   runs a real forward+backward pass with a synthetic image and text_feats
   batches of increasing size N, reporting peak CUDA memory per N -- shows
   the largest N that fits before OOM on the current GPU.

Doesn't load the real text encoder backbone -- text_hidden_dim comes from
AutoConfig.from_pretrained (a fast local config.json read), matching
train.py's own load_text_backbone=not data.embedding_cache path. The frozen
8B backbone stays loaded once regardless of N in real training, so it isn't
part of the per-step cost that actually scales with max_findings_per_image;
excluding it isolates just the part this flag controls.

Usage
-----
    python estimate_max_findings_per_image.py --config configs/default.yaml \\
        --override data.group_by_image=true data.multi_window=true \\
          data.spatial_size=[192,192,192] model.fusion_type=gated_cross_attention \\
          model.unet_base_channels=16 \\
          data.image_dir=/mnt/.../nifti_resampled_192 data.mask_dir=/mnt/.../labels_resampled_192

    # Skip the (slow, needs a GPU) memory probe and just see the data histogram:
    python estimate_max_findings_per_image.py --config configs/default.yaml --skip-memory-probe
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
# models.grounder/transformers are only needed for --skip-memory-probe=false
# (imported lazily inside build_model/probe_memory) -- keeps --skip-memory-
# probe usable without a working transformers install.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", metavar="KEY=VALUE", default=[],
                         help="Dot-notation config overrides -- pass the SAME overrides your real "
                              "training run uses, so the memory probe matches it exactly")
    parser.add_argument("--manifest", default=None,
                         help="Manifest to compute the findings-per-image histogram from "
                              "(default: data.train_manifest from --config)")
    parser.add_argument("--skip-histogram", action="store_true")
    parser.add_argument("--skip-memory-probe", action="store_true")
    parser.add_argument("--ns", type=int, nargs="*", default=[1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
                         help="Finding counts (N) to probe peak memory at")
    parser.add_argument("--text-len", type=int, default=None,
                         help="Synthetic text_feats sequence length (default: data.max_text_len -- "
                              "the padded worst case, for a conservative/safe estimate)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        key, _, raw_val = kv.partition("=")
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = yaml.safe_load(raw_val)
    return cfg


def print_histogram(manifest_path: str) -> None:
    with open(manifest_path) as f:
        samples = json.load(f)
    groups: dict[str, int] = {}
    for s in samples:
        groups[s["image"]] = groups.get(s["image"], 0) + 1
    counts = np.array(list(groups.values()))

    print("=" * 60)
    print(f"Findings-per-image histogram  ({manifest_path})")
    print("=" * 60)
    print(f"  unique images: {len(counts)}   total findings: {counts.sum()}")
    print(f"  min={counts.min()}  mean={counts.mean():.1f}  "
          f"p50={np.percentile(counts, 50):.0f}  p90={np.percentile(counts, 90):.0f}  "
          f"p95={np.percentile(counts, 95):.0f}  p99={np.percentile(counts, 99):.0f}  "
          f"max={counts.max()}")
    print()
    print("  cap N -> % of images truncated (findings beyond N dropped that draw)")
    for n in (4, 8, 12, 16, 24, 32, 48, 64):
        pct = (counts > n).mean() * 100
        print(f"    N={n:<4d} {pct:5.1f}% of images have more than N findings")
    print()


def build_model(cfg: dict, device: torch.device) -> "Grounder":
    from models.grounder import Grounder

    model = Grounder(
        text_encoder_name=cfg["model"]["text_encoder_name"],
        text_proj_dim=cfg["model"]["text_proj_dim"],
        freeze_text_encoder=cfg["model"]["freeze_text_encoder"],
        finetune_last_n_layers=cfg["model"].get("finetune_last_n_layers", 0),
        unet_base_channels=cfg["model"]["unet_base_channels"],
        unet_channel_mult=cfg["model"]["unet_channel_mult"],
        num_heads=cfg["model"]["num_heads"],
        target_q_tokens=cfg["model"]["target_q_tokens"],
        dropout=cfg["model"]["dropout"],
        in_channels=3 if cfg["data"].get("multi_window") else 1,
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        load_text_backbone=not cfg["data"].get("embedding_cache"),
        fusion_type=cfg["model"].get("fusion_type", "cross_attention"),
        voxtell_guidance_dim=cfg["model"].get("voxtell", {}).get("guidance_dim", 32),
        voxtell_prompt_decoder_dim=cfg["model"].get("voxtell", {}).get("prompt_decoder_dim", 256),
        voxtell_prompt_decoder_layers=cfg["model"].get("voxtell", {}).get("prompt_decoder_layers", 6),
        voxtell_prompt_decoder_heads=cfg["model"].get("voxtell", {}).get("prompt_decoder_heads", 8),
        freeze_language_projectors=cfg["model"].get("freeze_language_projectors", True),
    ).to(device)
    model.unet.grad_checkpointing = cfg["training"].get("gradient_checkpointing", False)
    return model


def probe_memory(cfg: dict, device: torch.device, ns: list[int], text_len: int) -> None:
    if device.type != "cuda":
        print("--device is not cuda; peak memory stats aren't meaningful, skipping probe.")
        return

    print("=" * 60)
    print("GPU memory probe")
    print("=" * 60)
    print(f"  device: {torch.cuda.get_device_name(device)}  "
          f"({torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GiB total)")

    from transformers import AutoConfig

    model = build_model(cfg, device)
    model.train()
    use_amp = cfg["training"].get("use_amp", True)

    text_hidden_dim = AutoConfig.from_pretrained(cfg["model"]["text_encoder_name"]).hidden_size
    channels = 3 if cfg["data"].get("multi_window") else 1
    spatial = tuple(cfg["data"]["spatial_size"])
    grad_clip = cfg["training"].get("grad_clip_norm", 1.0)

    print(f"  image: (1, {channels}, {spatial[0]}, {spatial[1]}, {spatial[2]})   "
          f"text_feats: (N, {text_len}, {text_hidden_dim})   amp={use_amp}   "
          f"grad_checkpointing={model.unet.grad_checkpointing}")
    print()
    print(f"  {'N':>5}  {'peak GiB':>10}")

    image = torch.randn(1, channels, *spatial, device=device)

    for n in sorted(ns):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.zero_grad(set_to_none=True)
        try:
            text_feats = torch.randn(n, text_len, text_hidden_dim, device=device)
            text_padding_mask = torch.zeros(n, text_len, dtype=torch.bool, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(image, text_feats=text_feats, text_padding_mask=text_padding_mask)
                loss = logits.float().pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
            print(f"  {n:>5}  {peak_gib:>10.2f}")
            del text_feats, text_padding_mask, logits, loss
        except torch.cuda.OutOfMemoryError:
            print(f"  {n:>5}  {'OOM':>10}")
            print(f"\n  Stopping -- N={n} OOM'd, larger N will too.")
            break

    print()
    print("  Pick a max_findings_per_image comfortably under the largest N that")
    print("  didn't OOM -- other processes/fragmentation eat into headroom this")
    print("  clean, empty-cache probe doesn't account for.")


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    if not args.skip_histogram:
        print_histogram(args.manifest or cfg["data"]["train_manifest"])

    if not args.skip_memory_probe:
        device = torch.device(args.device)
        text_len = args.text_len or cfg["data"]["max_text_len"]
        probe_memory(cfg, device, args.ns, text_len)


if __name__ == "__main__":
    main()
