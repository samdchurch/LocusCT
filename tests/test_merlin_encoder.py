import pytest
import torch

pytest.importorskip("merlin")

import torchvision  # noqa: E402
from merlin.models.i3res import I3ResNet  # noqa: E402

from models.merlin_encoder import MerlinEncoder  # noqa: E402

# Matches configs/default.yaml's default ch = [base_channels * m for m in
# channel_mult] with unet_base_channels=16, unet_channel_mult=[1,2,4,8,16].
OUT_CHANNELS = [16, 32, 64, 128, 256]


def _build_test_i3resnet() -> I3ResNet:
    """A real (untrained) I3ResNet152 -- exercises the actual upstream
    architecture (catches future merlin-vlm stride/kernel drift) without
    needing network access, a staged checkpoint, or Clinical-Longformer."""
    resnet2d = torchvision.models.resnet152(weights=None)
    return I3ResNet(resnet2d, conv_class=True, ImageEmbedding=True)


def _build_test_encoder(**kwargs) -> MerlinEncoder:
    return MerlinEncoder(out_channels=OUT_CHANNELS, _i3_resnet=_build_test_i3resnet(), **kwargs)


def test_tap_shapes() -> None:
    """Full-scale forward pass -- Merlin's architecture isn't parameterizable
    by input size the way the plain UNet path is, so this is inherently
    slower than the rest of the test suite (real ResNet152-3D at 224x224x160
    on CPU), but catches upstream stride/kernel drift early."""
    encoder = _build_test_encoder()
    encoder.eval()
    image = torch.randn(1, 1, *MerlinEncoder.FULL_GRID_HWD)

    with torch.no_grad():
        taps = encoder(image)

    for i, (key, expected_hwd) in enumerate(MerlinEncoder.TAP_SHAPES_HWD.items()):
        out = taps[i]
        assert out.shape[0] == 1
        assert out.shape[1] == OUT_CHANNELS[i]
        assert tuple(out.shape[2:]) == expected_hwd, f"{key}: {tuple(out.shape[2:])} != {expected_hwd}"


def test_frozen_by_default() -> None:
    encoder = _build_test_encoder()
    for name, p in encoder.named_parameters():
        if name.startswith("proj_"):
            assert p.requires_grad, f"{name} (projection adapter) should always be trainable"
        else:
            assert not p.requires_grad, f"{name} (Merlin backbone) should be frozen by default"


def test_finetune_last_n_stages_unfreezes_only_those_stages() -> None:
    encoder = _build_test_encoder(finetune_last_n_stages=2)

    assert all(p.requires_grad for p in encoder.layer4.parameters())
    assert all(p.requires_grad for p in encoder.layer3.parameters())
    assert not any(p.requires_grad for p in encoder.layer2.parameters())
    assert not any(p.requires_grad for p in encoder.layer1.parameters())
    assert not any(p.requires_grad for p in encoder.conv1.parameters())  # stem never unfrozen by this flag
