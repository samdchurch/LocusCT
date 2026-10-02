import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from .blocks import ResBlock3D
from .cross_attention import CrossAttentionFusion, GatedCrossAttentionFusion
from .voxtell_fusion import PromptDecoder, VoxTellDecoderBlock, masked_mean_pool


# ---------------------------------------------------------------------------
# Encoder block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Stride-2 ResBlock to downsample, then another ResBlock at same resolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResBlock3D(in_ch, out_ch, stride=2),
            ResBlock3D(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Decoder block with cross-attention fusion
# ---------------------------------------------------------------------------

class PlainDecoderBlock(nn.Module):
    """Upsample → concat skip → conv fusion (no text attention)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv_fuse = nn.Sequential(
            ResBlock3D(in_ch + skip_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_fuse(x)


class DecoderBlock(nn.Module):
    """
    Upsample → concat skip → convolutional fusion → cross-attention with text.
    Trilinear upsample (not ConvTranspose3d) to avoid checkerboard artifacts.
    F.interpolate(..., size=skip.shape[2:]) handles non-power-of-2 spatial dims.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        text_hidden_dim: int,
        text_proj_dim: int = 256,
        num_heads: int = 8,
        spatial_pool_stride: int = 1,
        dropout: float = 0.1,
        freeze_language_projectors: bool = True,
    ) -> None:
        super().__init__()
        self.conv_fuse = nn.Sequential(
            ResBlock3D(in_ch + skip_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )
        self.cross_attn = CrossAttentionFusion(
            image_dim=out_ch,
            text_dim=text_hidden_dim,
            proj_dim=text_proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            spatial_pool_stride=spatial_pool_stride,
            freeze_language_projectors=freeze_language_projectors,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        text_feats: torch.Tensor,
        text_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_fuse(x)
        x = self.cross_attn(x, text_feats, text_padding_mask)
        return x


class GatedDecoderBlock(nn.Module):
    """
    Upsample decoder stream → text-gated cross-attention (GatedCrossAttentionFusion,
    Q from the decoder stream itself) → THEN concat encoder skip → conv fusion.

    Matches the reference "Cross Attention Module" diagram, where the skip
    connection joins after the attention gate rather than before it -- unlike
    DecoderBlock (this file's default), which concats the skip first and only
    then cross-attends on the already-skip-fused representation.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        text_hidden_dim: int,
        text_proj_dim: int = 256,
        num_heads: int = 8,
        spatial_pool_stride: int = 1,
        dropout: float = 0.1,
        freeze_language_projectors: bool = True,
    ) -> None:
        super().__init__()
        self.gated_attn = GatedCrossAttentionFusion(
            image_dim=in_ch,
            text_dim=text_hidden_dim,
            proj_dim=text_proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            spatial_pool_stride=spatial_pool_stride,
            freeze_language_projectors=freeze_language_projectors,
        )
        self.conv_fuse = nn.Sequential(
            ResBlock3D(in_ch + skip_ch, out_ch),
            ResBlock3D(out_ch, out_ch),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        text_feats: torch.Tensor,
        text_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.gated_attn(x, text_feats, text_padding_mask)
        x = torch.cat([x, skip], dim=1)
        return self.conv_fuse(x)


# ---------------------------------------------------------------------------
# Full 3D UNet
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    3D UNet with cross-attention text fusion in each decoder level.

    Channel progression (base_channels=32, channel_mult=(1,2,4,8,16)):
      enc0: 32   (stride-1 init conv)
      enc1: 32   (stride-2)
      enc2: 64   (stride-2)
      enc3: 128  (stride-2)
      enc4: 256  (stride-2)
      bottleneck: 512
      dec4: 256  (skip=enc3)
      dec3: 128  (skip=enc2)
      dec2: 64   (skip=enc1)
      dec1: 32   (skip=enc0)
      output: 1

    encoder_type="merlin" swaps this plain encoder for a frozen pretrained
    Merlin I3ResNet152 (see models/merlin_encoder.py::MerlinEncoder) --
    the decoder (bottleneck, dec4..dec1, all fusion modules, head) is
    completely unchanged either way, since it only ever consumes 5
    pre-declared-channel-count tensors (e0..e4) and upsamples to whatever
    spatial size each skip tensor happens to have.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8, 16),
        text_hidden_dim: int = 256,
        text_proj_dim: int = 256,
        num_heads: int = 8,
        target_q_tokens: int = 2048,
        dropout: float = 0.1,
        spatial_size: tuple[int, int, int] = (352, 352, 192),
        fusion_type: str = "cross_attention",
        voxtell_guidance_dim: int = 32,
        voxtell_prompt_decoder_dim: int = 256,
        voxtell_prompt_decoder_layers: int = 6,
        voxtell_prompt_decoder_heads: int = 8,
        freeze_language_projectors: bool = True,
        encoder_type: str = "unet",
        merlin_model_dir: str = "",
        merlin_clinical_longformer_dir: str = "",
        freeze_merlin_encoder: bool = True,
        finetune_last_n_merlin_stages: int = 0,
    ) -> None:
        super().__init__()
        if fusion_type not in ("cross_attention", "voxtell", "encoder_cross_attention", "gated_cross_attention"):
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        if encoder_type not in ("unet", "merlin"):
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        self.fusion_type = fusion_type
        self.encoder_type = encoder_type
        ch = [base_channels * m for m in channel_mult]  # [32, 64, 128, 256, 512]

        # --- Encoder ---
        if encoder_type == "unet":
            self.init_conv = nn.Sequential(
                ResBlock3D(in_channels, ch[0]),
                ResBlock3D(ch[0], ch[0]),
            )
            self.enc1 = EncoderBlock(ch[0], ch[1])
            self.enc2 = EncoderBlock(ch[1], ch[2])
            self.enc3 = EncoderBlock(ch[2], ch[3])
            self.enc4 = EncoderBlock(ch[3], ch[4])
        else:  # "merlin"
            from .merlin_encoder import MerlinEncoder

            self.merlin_encoder = MerlinEncoder(
                out_channels=ch,
                model_dir=merlin_model_dir,
                clinical_longformer_dir=merlin_clinical_longformer_dir,
                freeze=freeze_merlin_encoder,
                finetune_last_n_stages=finetune_last_n_merlin_stages,
            )

        # --- Bottleneck ---
        bottleneck_ch = ch[4] * 2  # 1024 would be large; use ch[4] itself as bottleneck
        # Actually use 2x the last encoder stage for bottleneck expressiveness:
        # ch = [32,64,128,256,512] so bottleneck = 512 (ch[4] already)
        # We treat ch[4] as both enc4 output AND bottleneck input/output for simplicity.
        # An extra bottleneck block at the same channel count:
        self.bottleneck = nn.Sequential(
            ResBlock3D(ch[4], ch[4]),
            ResBlock3D(ch[4], ch[4]),
        )

        if fusion_type == "voxtell":
            G = voxtell_guidance_dim
            # VoxTell pools text to one query vector and decodes per-stage guidance from
            # it (see PromptDecoder.stage_adapters) rather than attending per-stage like
            # CrossAttentionFusion, so it keeps its single q_proj -- just sized to the raw
            # hidden dim now instead of a shared pre-projected one.
            self.prompt_decoder = PromptDecoder(
                text_feat_dim=text_hidden_dim,
                bottleneck_ch=ch[4],
                stage_channels=[ch[3], ch[2], ch[1], ch[0]],
                guidance_dim=G,
                hidden_dim=voxtell_prompt_decoder_dim,
                num_layers=voxtell_prompt_decoder_layers,
                num_heads=voxtell_prompt_decoder_heads,
                dropout=dropout,
            )
            # Each stage's input carries the previous stage's +G guidance channels
            # (paper eq. 4: y^up_{s-1}), except dec4 whose input is the bottleneck.
            self.dec4 = VoxTellDecoderBlock(ch[4],     ch[3], ch[3])
            self.dec3 = VoxTellDecoderBlock(ch[3] + G, ch[2], ch[2])
            self.dec2 = VoxTellDecoderBlock(ch[2] + G, ch[1], ch[1])
            self.dec1 = VoxTellDecoderBlock(ch[1] + G, ch[0], ch[0])
            self.head = nn.Conv3d(ch[0] + G, 1, kernel_size=1)
        elif fusion_type == "encoder_cross_attention":
            # Cross-attention applied after each encoder level; plain decoder uses
            # the text-fused skip connections without additional fusion.
            pool_strides = self._compute_pool_strides(spatial_size, target_q_tokens)
            # _compute_pool_strides returns [s_D/16, s_D/8, s_D/4, s_D/2];
            # reverse to get encoder order [enc1=D/2, enc2=D/4, enc3=D/8, enc4=D/16].
            enc_strides = pool_strides[::-1]
            self.enc_attns = nn.ModuleList([
                CrossAttentionFusion(ch[1], text_hidden_dim, text_proj_dim, num_heads, dropout, enc_strides[0],
                                      freeze_language_projectors=freeze_language_projectors),
                CrossAttentionFusion(ch[2], text_hidden_dim, text_proj_dim, num_heads, dropout, enc_strides[1],
                                      freeze_language_projectors=freeze_language_projectors),
                CrossAttentionFusion(ch[3], text_hidden_dim, text_proj_dim, num_heads, dropout, enc_strides[2],
                                      freeze_language_projectors=freeze_language_projectors),
                CrossAttentionFusion(ch[4], text_hidden_dim, text_proj_dim, num_heads, dropout, enc_strides[3],
                                      freeze_language_projectors=freeze_language_projectors),
            ])
            self.dec4 = PlainDecoderBlock(ch[4], ch[3], ch[3])
            self.dec3 = PlainDecoderBlock(ch[3], ch[2], ch[2])
            self.dec2 = PlainDecoderBlock(ch[2], ch[1], ch[1])
            self.dec1 = PlainDecoderBlock(ch[1], ch[0], ch[0])
            self.head = nn.Conv3d(ch[0], 1, kernel_size=1)
        else:
            # cross_attention (skip concatenated before attention, additive residual)
            # or gated_cross_attention (skip concatenated after attention, tanh-gated
            # multiplicative combine) -- same wiring, just a different block class.
            block_cls = GatedDecoderBlock if fusion_type == "gated_cross_attention" else DecoderBlock

            # --- Compute pool strides per decoder level ---
            # After 4 downsamples: level-4 spatial = (D/16, H/16, W/16)
            pool_strides = self._compute_pool_strides(spatial_size, target_q_tokens)

            # --- Decoder ---
            # dec4: in=ch[4] (bottleneck), skip=ch[3] (enc3), out=ch[3]
            self.dec4 = block_cls(ch[4], ch[3], ch[3], text_hidden_dim, text_proj_dim, num_heads, pool_strides[0], dropout,
                                   freeze_language_projectors)
            # dec3: in=ch[3], skip=ch[2] (enc2), out=ch[2]
            self.dec3 = block_cls(ch[3], ch[2], ch[2], text_hidden_dim, text_proj_dim, num_heads, pool_strides[1], dropout,
                                   freeze_language_projectors)
            # dec2: in=ch[2], skip=ch[1] (enc1), out=ch[1]
            self.dec2 = block_cls(ch[2], ch[1], ch[1], text_hidden_dim, text_proj_dim, num_heads, pool_strides[2], dropout,
                                   freeze_language_projectors)
            # dec1: in=ch[1], skip=ch[0] (init_conv), out=ch[0]
            self.dec1 = block_cls(ch[1], ch[0], ch[0], text_hidden_dim, text_proj_dim, num_heads, pool_strides[3], dropout,
                                   freeze_language_projectors)

            # --- Output head ---
            self.head = nn.Conv3d(ch[0], 1, kernel_size=1)

        # Set by train.py when training.gradient_checkpointing=true.
        self.grad_checkpointing = False

    def _ckpt(self, module: nn.Module, *args: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpointing and self.training:
            return ckpt.checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    @staticmethod
    def _compute_pool_strides(
        spatial_size: tuple[int, int, int],
        target_q_tokens: int,
    ) -> list[int]:
        """
        For each of the 4 decoder levels (from deep to shallow),
        compute avg-pool stride s such that pooled N ≈ target_q_tokens.
        Level k spatial size = (D/2^k, H/2^k, W/2^k) for k in [4, 3, 2, 1].
        """
        D, H, W = spatial_size
        strides = []
        for k in [4, 3, 2, 1]:
            d = max(1, D // (2 ** k))
            h = max(1, H // (2 ** k))
            w = max(1, W // (2 ** k))
            n = d * h * w
            s = max(1, round((n / target_q_tokens) ** (1.0 / 3.0)))
            strides.append(s)
        return strides

    def forward(
        self,
        image: torch.Tensor,                       # (B_img, 1, D, H, W)
        text_feats: torch.Tensor,                  # (B_txt, L, text_hidden_dim) -- raw, unprojected
        text_padding_mask: torch.Tensor | None = None,  # (B_txt, L)
    ) -> torch.Tensor:
        """
        B_img must equal B_txt (the normal case, including all val/eval call
        sites), OR B_img must be 1 with B_txt > 1 -- the shared-encoder
        training mode (see data.dataset.GrounderImageGroupedDataset): one
        image, N referring expressions. In that mode the encoder/bottleneck
        below run ONCE at batch 1 and their outputs are broadcast to N right
        before the text-conditioned decoder, which still runs N times.
        """
        B_img, B_txt = image.shape[0], text_feats.shape[0]
        if B_img != B_txt and B_img != 1:
            raise ValueError(f"image batch ({B_img}) must equal text batch ({B_txt}) or be 1")
        expand_needed = B_img != B_txt

        if self.fusion_type == "encoder_cross_attention" and expand_needed:
            # This fusion_type fuses text in at e1 (before the bottleneck), so
            # there's no meaningful shared-encoder benefit to broadcast into --
            # just expand the image itself and run the encoder at batch B_txt
            # like the non-shared case (correctness fallback, not an optimization).
            image = image.expand(B_txt, *image.shape[1:]).contiguous()

        # Encoder
        if self.encoder_type == "unet":
            e0 = self._ckpt(self.init_conv, image)  # (B, ch[0], D,    H,    W)
            e1 = self._ckpt(self.enc1, e0)          # (B, ch[1], D/2,  H/2,  W/2)
            e2 = self._ckpt(self.enc2, e1)          # (B, ch[2], D/4,  H/4,  W/4)
            e3 = self._ckpt(self.enc3, e2)          # (B, ch[3], D/8,  H/8,  W/8)
            e4 = self._ckpt(self.enc4, e3)          # (B, ch[4], D/16, H/16, W/16)
        else:  # "merlin" -- MerlinEncoder returns all 5 channel-projected taps at once
            e0, e1, e2, e3, e4 = self._ckpt(self.merlin_encoder, image)

        if self.fusion_type == "encoder_cross_attention":
            e1 = self._ckpt(self.enc_attns[0], e1, text_feats, text_padding_mask)
            e2 = self._ckpt(self.enc_attns[1], e2, text_feats, text_padding_mask)
            e3 = self._ckpt(self.enc_attns[2], e3, text_feats, text_padding_mask)
            e4 = self._ckpt(self.enc_attns[3], e4, text_feats, text_padding_mask)

        # Bottleneck
        b = self._ckpt(self.bottleneck, e4)     # (B, ch[4], D/16, H/16, W/16)

        if expand_needed and self.fusion_type != "encoder_cross_attention":
            # Encoder ran once at batch 1 above -- broadcast its outputs to
            # every decoder-fusion module below, none of which support
            # mismatched image/text batch dims on their own (CrossAttentionFusion/
            # GatedCrossAttentionFusion reshape Q/K/V off image_feats' own batch;
            # PromptDecoder's nn.TransformerDecoder needs tgt/memory batch to match).
            e0, e1, e2, e3, e4, b = (
                t.expand(B_txt, *t.shape[1:]).contiguous() for t in (e0, e1, e2, e3, e4, b)
            )

        if self.fusion_type == "voxtell":
            q = masked_mean_pool(text_feats, text_padding_mask)      # (B, text_hidden_dim)
            T4, T3, T2, T1 = self.prompt_decoder(q, b)                # T_s: (B, G, ch[s])
            d4 = self._ckpt(self.dec4, b,  e3, T4)   # (B, ch[3]+G, D/8,  ...)
            d3 = self._ckpt(self.dec3, d4, e2, T3)   # (B, ch[2]+G, D/4,  ...)
            d2 = self._ckpt(self.dec2, d3, e1, T2)   # (B, ch[1]+G, D/2,  ...)
            d1 = self._ckpt(self.dec1, d2, e0, T1)   # (B, ch[0]+G, D,    ...)
        elif self.fusion_type == "encoder_cross_attention":
            d4 = self._ckpt(self.dec4, b,  e3)       # (B, ch[3], D/8,  ...)
            d3 = self._ckpt(self.dec3, d4, e2)       # (B, ch[2], D/4,  ...)
            d2 = self._ckpt(self.dec2, d3, e1)       # (B, ch[1], D/2,  ...)
            d1 = self._ckpt(self.dec1, d2, e0)       # (B, ch[0], D,    ...)
        else:
            # Decoder with cross-attention (cross_attention or gated_cross_attention)
            d4 = self._ckpt(self.dec4, b,  e3, text_feats, text_padding_mask)  # (B, ch[3], D/8,  ...)
            d3 = self._ckpt(self.dec3, d4, e2, text_feats, text_padding_mask)  # (B, ch[2], D/4,  ...)
            d2 = self._ckpt(self.dec2, d3, e1, text_feats, text_padding_mask)  # (B, ch[1], D/2,  ...)
            d1 = self._ckpt(self.dec1, d2, e0, text_feats, text_padding_mask)  # (B, ch[0], D,    ...)

        if self.encoder_type == "merlin":
            # dec1's skip (e0) is Merlin's shallowest tap, only H,W-halved
            # relative to the target grid -- D already matches exactly, since
            # conv1's inflated stride (1,2,2) never touches D (see
            # MerlinEncoder's docstring). One trilinear upsample closes the
            # remaining H,W gap; safe/geometrically-consistent since it stays
            # within Merlin's own single working grid.
            d1 = F.interpolate(d1, size=self.merlin_encoder.FULL_GRID_HWD, mode="trilinear", align_corners=False)

        return self.head(d1)          # (B, 1, D, H, W)
