#!/usr/bin/env python3
"""
Throwaway smoke test for the new fusion_type="voxtell" path (and a regression
check on fusion_type="cross_attention"). Bypasses the real text encoder via
load_text_backbone=False + precomputed text_feats, so it needs no HF weights
or network access -- just torch. Run on a machine/container with torch+CUDA
or CPU:

    python smoke_test_voxtell_fusion.py
"""

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from models.grounder import Grounder

B, D, H, W = 2, 32, 32, 32
TEXT_FEAT_DIM, L = 16, 5


def run(fusion_type: str, grad_checkpointing: bool) -> None:
    model = Grounder(
        text_hidden_dim=TEXT_FEAT_DIM,
        load_text_backbone=False,
        unet_base_channels=4,
        unet_channel_mult=(1, 2, 4, 8, 16),
        num_heads=2,
        target_q_tokens=64,
        dropout=0.0,
        spatial_size=(D, H, W),
        fusion_type=fusion_type,
        voxtell_guidance_dim=4,
        voxtell_prompt_decoder_dim=16,
        voxtell_prompt_decoder_layers=1,
        voxtell_prompt_decoder_heads=2,
    )
    model.unet.grad_checkpointing = grad_checkpointing
    model.train()

    image = torch.randn(B, 1, D, H, W, requires_grad=True)
    text_feats = torch.randn(B, L, TEXT_FEAT_DIM)
    text_padding_mask = torch.zeros(B, L, dtype=torch.bool)
    text_padding_mask[:, -1] = True  # last token is padding, for the mean-pool path

    logits = model(image, text_feats=text_feats, text_padding_mask=text_padding_mask)
    assert logits.shape == (B, 1, D, H, W), f"shape mismatch: {logits.shape}"

    logits.sum().backward()
    assert image.grad is not None, "no gradient reached the input image"
    n_params_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    print(
        f"[OK] fusion_type={fusion_type!r} grad_checkpointing={grad_checkpointing} "
        f"-> logits {tuple(logits.shape)}, grads on {n_params_with_grad}/{n_trainable} trainable params"
    )


if __name__ == "__main__":
    for fusion_type in ("cross_attention", "voxtell", "gated_cross_attention"):
        for grad_checkpointing in (False, True):
            run(fusion_type, grad_checkpointing)
    print("All smoke tests passed.")
