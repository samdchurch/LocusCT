import pytest
import torch

from models.unet3d import UNet3D

FUSION_TYPES = ["cross_attention", "gated_cross_attention", "voxtell", "encoder_cross_attention"]

# Small enough to be fast, divisible by 16 (4 downsample stages need integer
# dims at every level), all-heads-divide-all-channel-counts (base_channels=4,
# channel_mult=(1,2,4,8,16) -> ch=[4,8,16,32,64], num_heads=2 divides all of them).
SPATIAL_SIZE = (32, 32, 32)
BASE_CHANNELS = 4
NUM_HEADS = 2
TEXT_HIDDEN_DIM = 16
TEXT_PROJ_DIM = 8


def _build_unet(fusion_type: str) -> UNet3D:
    return UNet3D(
        in_channels=1,
        base_channels=BASE_CHANNELS,
        channel_mult=(1, 2, 4, 8, 16),
        text_hidden_dim=TEXT_HIDDEN_DIM,
        text_proj_dim=TEXT_PROJ_DIM,
        num_heads=NUM_HEADS,
        target_q_tokens=2048,
        dropout=0.0,
        spatial_size=SPATIAL_SIZE,
        fusion_type=fusion_type,
        voxtell_guidance_dim=4,
        voxtell_prompt_decoder_dim=16,
        voxtell_prompt_decoder_layers=1,
        voxtell_prompt_decoder_heads=2,
    )


def _random_text(n: int, length: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    text_feats = torch.randn(n, length, TEXT_HIDDEN_DIM)
    text_padding_mask = torch.zeros(n, length, dtype=torch.bool)  # no padding
    return text_feats, text_padding_mask


@pytest.mark.parametrize("fusion_type", FUSION_TYPES)
@pytest.mark.parametrize("n", [1, 3])
def test_forward_shape_and_gradients(fusion_type: str, n: int) -> None:
    """image batch 1, text batch n -- covers both the untouched B_img==B_txt
    path (n=1) and the new shared-encoder broadcast path (n=3)."""
    model = _build_unet(fusion_type)
    image = torch.randn(1, 1, *SPATIAL_SIZE)
    text_feats, text_padding_mask = _random_text(n)

    logits = model(image, text_feats, text_padding_mask)
    assert logits.shape == (n, 1, *SPATIAL_SIZE)

    logits.sum().backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"


@pytest.mark.parametrize("fusion_type", FUSION_TYPES)
def test_mismatched_batch_raises(fusion_type: str) -> None:
    model = _build_unet(fusion_type)
    image = torch.randn(2, 1, *SPATIAL_SIZE)
    text_feats, text_padding_mask = _random_text(3)

    with pytest.raises(ValueError):
        model(image, text_feats, text_padding_mask)


def test_merlin_encoder_forward_shape_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    """encoder_type="merlin": full-scale forward+backward (Merlin's fixed
    224x224x160 grid, not the tiny SPATIAL_SIZE used above -- its
    architecture isn't parameterizable by input size), monkeypatching
    load_merlin_i3resnet so this needs no staged checkpoint/Clinical-
    Longformer/network access -- just the real (untrained) upstream
    I3ResNet152 architecture, to also catch future merlin-vlm drift."""
    pytest.importorskip("merlin")
    import torchvision
    from merlin.models.i3res import I3ResNet

    from models.merlin_encoder import MerlinEncoder

    monkeypatch.setattr(
        "models.merlin_encoder.load_merlin_i3resnet",
        lambda model_dir, clinical_longformer_dir: I3ResNet(
            torchvision.models.resnet152(weights=None), conv_class=True, ImageEmbedding=True
        ),
    )

    model = UNet3D(
        in_channels=1,
        base_channels=16,  # matches configs/default.yaml's default -> ch=[16,32,64,128,256]
        channel_mult=(1, 2, 4, 8, 16),
        text_hidden_dim=TEXT_HIDDEN_DIM,
        text_proj_dim=TEXT_PROJ_DIM,
        num_heads=NUM_HEADS,
        target_q_tokens=2048,
        dropout=0.0,
        spatial_size=MerlinEncoder.FULL_GRID_HWD,
        fusion_type="gated_cross_attention",
        encoder_type="merlin",
        merlin_model_dir="unused",  # bootstrap is monkeypatched away above
        merlin_clinical_longformer_dir="unused",
    )
    image = torch.randn(1, 1, *MerlinEncoder.FULL_GRID_HWD)
    text_feats, text_padding_mask = _random_text(1)

    logits = model(image, text_feats, text_padding_mask)
    assert logits.shape == (1, 1, *MerlinEncoder.FULL_GRID_HWD)

    logits.sum().backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"

    # Frozen-by-default Merlin backbone should have no gradient at all (not just
    # be absent from `missing` above, which only checks trainable params).
    frozen_backbone = [
        p for name, p in model.merlin_encoder.named_parameters() if not name.startswith("proj_")
    ]
    assert frozen_backbone  # sanity: the backbone actually has non-proj params
    assert all(not p.requires_grad and p.grad is None for p in frozen_backbone)
