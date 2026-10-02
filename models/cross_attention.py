import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Fuses 3D image features with text token features via multi-head cross-attention.

    Each instance owns its own text_proj (Linear(text_dim, proj_dim) + LayerNorm) --
    there is no shared projection upstream. text_feats arrives as the text
    encoder's raw, unprojected hidden states (same for every stage); this
    module projects them down to proj_dim itself, as its first step, before
    the existing k_proj/v_proj. Since text_proj lives here, inside the UNet,
    it trains normally regardless of embedding_cache -- unlike a shared
    projection living in the (possibly backbone-less, in cache mode)
    TextEncoder, it's never skipped or excluded from training.

    To handle large 3D spatial extents efficiently, image features are spatially
    average-pooled before forming queries (reducing N by stride^3), then the
    attention output is trilinearly upsampled back and added as a residual.

    forward():
        image_feats:      (B, C, D, H, W)
        text_feats:       (B, L, text_dim) -- raw text encoder hidden states
        text_padding_mask (B, L) bool — True at padding positions
    Returns: (B, C, D, H, W)
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        proj_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        spatial_pool_stride: int = 1,
        freeze_language_projectors: bool = True,
    ) -> None:
        super().__init__()
        assert image_dim % num_heads == 0, "image_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads
        self.spatial_pool_stride = spatial_pool_stride
        self.dropout_p = dropout

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        if freeze_language_projectors:
            # Defaults to frozen -- pairs naturally with train.py's --init_text_proj,
            # which warm-starts this exact module from a prior run's weights. Set
            # model.freeze_language_projectors=false in config to train it instead.
            for p in self.text_proj.parameters():
                p.requires_grad_(False)

        self.q_proj = nn.Linear(image_dim, image_dim)
        self.k_proj = nn.Linear(proj_dim, image_dim)
        self.v_proj = nn.Linear(proj_dim, image_dim)
        self.out_proj = nn.Linear(image_dim, image_dim)

        self.norm_img = nn.LayerNorm(image_dim)
        self.norm_txt = nn.LayerNorm(proj_dim)

    def forward(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, D, H, W = image_feats.shape
        s = self.spatial_pool_stride

        # --- Spatial pooling to keep Q tokens manageable ---
        if s > 1:
            pooled = F.avg_pool3d(image_feats, kernel_size=s, stride=s)
        else:
            pooled = image_feats
        pD, pH, pW = pooled.shape[2:]

        # Flatten spatial → token sequence: (B, N, C)
        img_tokens = pooled.flatten(2).transpose(1, 2)
        img_tokens = self.norm_img(img_tokens)

        text_feats = self.text_proj(text_feats)
        text_feats_n = self.norm_txt(text_feats)

        # Project Q, K, V
        Q = self.q_proj(img_tokens)       # (B, N, C)
        K = self.k_proj(text_feats_n)     # (B, L, C)
        V = self.v_proj(text_feats_n)     # (B, L, C)

        # Reshape to (B, heads, seq, head_dim)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Build additive attention bias for padding: -inf at pad positions
        attn_bias = None
        if text_padding_mask is not None:
            L = text_feats.size(1)
            attn_bias = torch.zeros(
                B, 1, 1, L, device=image_feats.device, dtype=image_feats.dtype
            )
            attn_bias = attn_bias.masked_fill(
                text_padding_mask[:, None, None, :], float("-inf")
            )

        # Scaled dot-product attention (Flash Attention path when available)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B, heads, N, head_dim)

        # Merge heads → (B, N, C)
        N = pD * pH * pW
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        attn_out = self.out_proj(attn_out)

        # Residual on pooled tokens, then reshape to spatial
        img_tokens = img_tokens + attn_out                      # (B, N, C) pre-norm residual
        spatial_out = img_tokens.transpose(1, 2).view(B, C, pD, pH, pW)

        # Upsample back to original spatial size
        if s > 1:
            spatial_out = F.interpolate(
                spatial_out, size=(D, H, W), mode="trilinear", align_corners=False
            )

        # Final additive residual onto original image features
        return image_feats + spatial_out


class GatedCrossAttentionFusion(nn.Module):
    """
    Text-conditioned attention gate, matching the "Cross Attention Module" reference
    diagram (RadBERT + per-stage Linear -> K/V, Q from the decoder stream itself,
    tanh(attention) gates the decoder stream via elementwise multiplication). Unlike
    CrossAttentionFusion (this file's default fusion, additive residual, skip
    concatenated *before* attention), here:
      - Q comes from the decoder stream passed in as image_feats (already upsampled,
        but NOT yet concatenated with the encoder skip -- see unet3d.GatedDecoderBlock,
        which concatenates the skip after this module runs).
      - The attention output is squashed through tanh into a [-1, 1] gate, then
        elementwise-multiplied back onto image_feats, instead of added as a residual.

    forward():
        image_feats:      (B, C, D, H, W) -- decoder stream, pre-skip-concat
        text_feats:       (B, L, text_dim) -- raw text encoder hidden states
        text_padding_mask (B, L) bool — True at padding positions
    Returns: (B, C, D, H, W)
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        proj_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        spatial_pool_stride: int = 1,
        freeze_language_projectors: bool = True,
    ) -> None:
        super().__init__()
        assert image_dim % num_heads == 0, "image_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = image_dim // num_heads
        self.spatial_pool_stride = spatial_pool_stride
        self.dropout_p = dropout

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )
        if freeze_language_projectors:
            for p in self.text_proj.parameters():
                p.requires_grad_(False)

        self.q_proj = nn.Linear(image_dim, image_dim)
        self.k_proj = nn.Linear(proj_dim, image_dim)
        self.v_proj = nn.Linear(proj_dim, image_dim)
        self.out_proj = nn.Linear(image_dim, image_dim)

        self.norm_img = nn.LayerNorm(image_dim)
        self.norm_txt = nn.LayerNorm(proj_dim)

    def forward(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, D, H, W = image_feats.shape
        s = self.spatial_pool_stride

        if s > 1:
            pooled = F.avg_pool3d(image_feats, kernel_size=s, stride=s)
        else:
            pooled = image_feats
        pD, pH, pW = pooled.shape[2:]

        # Flatten spatial → token sequence: (B, N, C). Kept separate from its
        # normalized copy below -- unlike CrossAttentionFusion's residual (which
        # can reuse the normalized tokens), the gate here must multiply onto the
        # raw decoder stream, not a LayerNorm'd copy of it.
        img_tokens = pooled.flatten(2).transpose(1, 2)
        img_tokens_n = self.norm_img(img_tokens)

        text_feats = self.text_proj(text_feats)
        text_feats_n = self.norm_txt(text_feats)

        Q = self.q_proj(img_tokens_n)     # (B, N, C)
        K = self.k_proj(text_feats_n)     # (B, L, C)
        V = self.v_proj(text_feats_n)     # (B, L, C)

        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_bias = None
        if text_padding_mask is not None:
            L = text_feats.size(1)
            attn_bias = torch.zeros(
                B, 1, 1, L, device=image_feats.device, dtype=image_feats.dtype
            )
            attn_bias = attn_bias.masked_fill(
                text_padding_mask[:, None, None, :], float("-inf")
            )

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # (B, heads, N, head_dim)

        N = pD * pH * pW
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        attn_out = self.out_proj(attn_out)

        gate = torch.tanh(attn_out)          # (B, N, C) in [-1, 1]
        gated = img_tokens * gate            # elementwise multiply onto the raw decoder stream
        spatial_out = gated.transpose(1, 2).view(B, C, pD, pH, pW)

        if s > 1:
            spatial_out = F.interpolate(
                spatial_out, size=(D, H, W), mode="trilinear", align_corners=False
            )

        return spatial_out
