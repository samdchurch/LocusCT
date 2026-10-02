import torch
import torch.nn as nn
from transformers import AutoConfig

from .text_encoder import TextEncoder
from .unet3d import UNet3D


class Grounder(nn.Module):
    """
    Top-level visual grounding model.

    Given a 3D CT volume and a referring expression, predicts a 3D binary
    segmentation mask localizing the referred region.

    forward() returns logits (B, 1, D, H, W) — apply sigmoid for probabilities
    or use BCEWithLogitsLoss directly during training.
    """

    def __init__(
        self,
        text_encoder_name: str = "Qwen/Qwen3-Embedding-8B",
        freeze_text_encoder: bool = True,
        finetune_last_n_layers: int = 0,
        unet_base_channels: int = 32,
        unet_channel_mult: tuple[int, ...] = (1, 2, 4, 8, 16),
        num_heads: int = 8,
        target_q_tokens: int = 2048,
        dropout: float = 0.1,
        in_channels: int = 1,
        spatial_size: tuple[int, int, int] = (352, 352, 192),
        load_text_backbone: bool = True,
        fusion_type: str = "cross_attention",
        voxtell_guidance_dim: int = 32,
        voxtell_prompt_decoder_dim: int = 256,
        voxtell_prompt_decoder_layers: int = 6,
        voxtell_prompt_decoder_heads: int = 8,
        text_proj_dim: int = 256,
        text_hidden_dim: int | None = None,
        freeze_language_projectors: bool = True,
        encoder_type: str = "unet",
        merlin_model_dir: str = "",
        merlin_clinical_longformer_dir: str = "",
        freeze_merlin_encoder: bool = True,
        finetune_last_n_merlin_stages: int = 0,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(
            model_name=text_encoder_name,
            freeze_backbone=freeze_text_encoder,
            finetune_last_n_layers=finetune_last_n_layers,
            load_backbone=load_text_backbone,
        )
        # Needed to size each fusion module's own text projection even when
        # load_text_backbone=False (embedding_cache mode never loads the
        # backbone weights, but the UNet's per-stage k_proj/v_proj/q_proj
        # still need to know the raw hidden size they're projecting from).
        # Pass text_hidden_dim explicitly to skip the AutoConfig lookup
        # entirely (e.g. synthetic dims in tests with no real model on disk).
        if text_hidden_dim is None:
            text_hidden_dim = AutoConfig.from_pretrained(text_encoder_name).hidden_size
        self.unet = UNet3D(
            in_channels=in_channels,
            base_channels=unet_base_channels,
            channel_mult=tuple(unet_channel_mult),
            text_hidden_dim=text_hidden_dim,
            text_proj_dim=text_proj_dim,
            num_heads=num_heads,
            target_q_tokens=target_q_tokens,
            dropout=dropout,
            spatial_size=tuple(spatial_size),
            fusion_type=fusion_type,
            voxtell_guidance_dim=voxtell_guidance_dim,
            voxtell_prompt_decoder_dim=voxtell_prompt_decoder_dim,
            voxtell_prompt_decoder_layers=voxtell_prompt_decoder_layers,
            voxtell_prompt_decoder_heads=voxtell_prompt_decoder_heads,
            freeze_language_projectors=freeze_language_projectors,
            encoder_type=encoder_type,
            merlin_model_dir=merlin_model_dir,
            merlin_clinical_longformer_dir=merlin_clinical_longformer_dir,
            freeze_merlin_encoder=freeze_merlin_encoder,
            finetune_last_n_merlin_stages=finetune_last_n_merlin_stages,
        )

    def forward(
        self,
        image: torch.Tensor,                        # (B, C, D, H, W)
        input_ids: torch.Tensor | None = None,      # (B, L)
        attention_mask: torch.Tensor | None = None, # (B, L)
        text_feats: torch.Tensor | None = None,     # (B, L, D) precomputed
        text_padding_mask: torch.Tensor | None = None,  # (B, L) bool precomputed
    ) -> torch.Tensor:
        """Returns logits (B, 1, D, H, W)."""
        if text_feats is None:
            text_feats, text_padding_mask = self.text_encoder(input_ids, attention_mask)
        return self.unet(image, text_feats, text_padding_mask)

    @torch.inference_mode()
    def predict(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Returns binary mask (B, 1, D, H, W) bool."""
        logits = self.forward(image, input_ids, attention_mask)
        return torch.sigmoid(logits) > threshold
