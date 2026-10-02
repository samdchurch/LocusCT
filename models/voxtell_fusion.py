import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResBlock3D


def masked_mean_pool(
    text_feats: torch.Tensor,
    text_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Pool per-token text features to a single embedding per prompt.

    text_feats:        (B, L, text_dim)
    text_padding_mask: (B, L) bool — True at padding positions
    Returns: (B, text_dim)
    """
    if text_padding_mask is None:
        return text_feats.mean(dim=1)
    valid = (~text_padding_mask).to(text_feats.dtype).unsqueeze(-1)  # (B, L, 1)
    summed = (text_feats * valid).sum(dim=1)
    count = valid.sum(dim=1).clamp(min=1.0)
    return summed / count


class PromptDecoder(nn.Module):
    """
    VoxTell's prompt decoder (paper eq. 2-3): a transformer with G learned query
    slots, conditioned on the pooled text embedding q, cross-attends over the
    flattened bottleneck image features to produce per-decoder-stage textual
    guidance tensors T_s in R^{G x C_s}.
    """

    def __init__(
        self,
        text_feat_dim: int,
        bottleneck_ch: int,
        stage_channels: list[int],
        guidance_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.guidance_dim = guidance_dim
        self.query_embed = nn.Parameter(torch.randn(guidance_dim, hidden_dim) * 0.02)
        self.q_proj = nn.Linear(text_feat_dim, hidden_dim)
        self.kv_proj = nn.Linear(bottleneck_ch, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.stage_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, c),
            )
            for c in stage_channels
        ])

    def forward(
        self,
        q: torch.Tensor,                  # (B, text_feat_dim) pooled text embedding
        bottleneck_feats: torch.Tensor,   # (B, C_bottleneck, D, H, W)
    ) -> list[torch.Tensor]:
        B = q.size(0)
        kv = bottleneck_feats.flatten(2).transpose(1, 2)  # (B, N, C_bottleneck)
        kv = self.kv_proj(kv)                              # (B, N, hidden_dim)

        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1) + self.q_proj(q).unsqueeze(1)
        out = self.transformer(tgt=queries, memory=kv)     # (B, G, hidden_dim)

        return [adapter(out) for adapter in self.stage_adapters]  # each (B, G, C_s)


class VoxTellDecoderBlock(nn.Module):
    """
    VoxTell's per-stage cross-scale fusion (paper eq. 4-5):
    upsample -> concat skip -> ConvBlock -> channel-wise dot product with the
    stage's textual guidance T_s -> concat the G resulting channels back on.

    forward() input/output both carry the +G guidance channels from the
    previous stage (per eq. 4's y^up_{s-1}), except the first stage, whose
    input is the bottleneck (no guidance channels yet).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv_fuse = nn.Sequential(
            ResBlock3D(in_ch + skip_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        guidance: torch.Tensor,  # (B, G, out_ch) -- T_s
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        z = self.conv_fuse(x)                                        # (B, out_ch, D, H, W) = z'_s
        dot = torch.einsum("bgc,bcdhw->bgdhw", guidance, z)           # (B, G, D, H, W)
        return torch.cat([z, dot], dim=1)                             # (B, out_ch + G, D, H, W) = y_s
