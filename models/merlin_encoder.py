import torch
import torch.nn as nn

from .merlin_utils import load_merlin_i3resnet


class MerlinEncoder(nn.Module):
    """
    Frozen(-by-default) Merlin I3ResNet152 encoder, tapped at 5 stages and
    channel-projected to feed UNet3D's e0..e4 encoder roles -- a drop-in
    replacement for UNet3D's own init_conv/enc1-4 path, letting the decoder
    (bottleneck, dec4..dec1, all fusion modules) run completely unchanged.

    Operates on Merlin's own fixed grid: input is always (B, 1, 224, 224, 160)
    (H, W, D), matching MerlinGridDataset's output -- NOT configurable, unlike
    the plain UNet3D encoder path, since Merlin's own preprocessing (RAS
    orientation, 1.5x1.5x3mm spacing) fixes this size.

    Merlin's I3ResNet has no true full-resolution tap (ResNets always
    downsample immediately) -- the shallowest available tap (post conv1) is
    only half-resolution in H,W (D is untouched, since conv1's inflated 7x7
    conv is hardcoded to stride (1,2,2) in merlin's own inflate_conv). UNet3D
    closes this gap with one final F.interpolate up to FULL_GRID_HWD after
    dec1, using this class's FULL_GRID_HWD constant.

    return_skips=True on Merlin's own I3ResNet.forward is unreachable when
    ImageEmbedding=True (the conv_class branch returns before ever checking
    it), so this class replicates the sequential submodule calls itself
    rather than relying on that flag.
    """

    # Merlin's fixed working grid (H, W, D) -- not derived from config.
    FULL_GRID_HWD = (224, 224, 160)

    # Tap shapes (H, W, D), i.e. permute(0,1,3,4,2) applied to each internal
    # (B,C,D,H,W) tap -- mirrors I3ResNet.forward's own
    # skips.append(x.permute(0,1,3,4,2)) convention (unreachable in
    # ImageEmbedding=True mode, hence replicated here). Derived from
    # inflate_conv's hardcoded stride=(1,2,2) for the 7x7 conv1 (never
    # reduces depth) and inflate_pool/Bottleneck3d's explicit stride-2 stages.
    TAP_SHAPES_HWD = {
        "e0": (112, 112, 160),  # post conv1/bn1/relu
        "e1": (56, 56, 80),     # post layer1
        "e2": (28, 28, 40),     # post layer2
        "e3": (14, 14, 20),     # post layer3
        "e4": (7, 7, 10),       # post layer4 (bottleneck input)
    }
    TAP_CHANNELS = {"e0": 64, "e1": 256, "e2": 512, "e3": 1024, "e4": 2048}

    def __init__(
        self,
        out_channels: list[int],  # [ch0..ch4], e.g. [16,32,64,128,256]
        model_dir: str = "",
        clinical_longformer_dir: str = "",
        freeze: bool = True,
        finetune_last_n_stages: int = 0,  # 0..4
        _i3_resnet: nn.Module | None = None,  # test-only escape hatch, bypasses load_merlin_i3resnet
    ) -> None:
        super().__init__()
        i3 = _i3_resnet if _i3_resnet is not None else load_merlin_i3resnet(model_dir, clinical_longformer_dir)
        self.conv1, self.bn1, self.relu, self.maxpool = i3.conv1, i3.bn1, i3.relu, i3.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = i3.layer1, i3.layer2, i3.layer3, i3.layer4

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            if finetune_last_n_stages > 0:
                self._unfreeze_last_n_stages(finetune_last_n_stages)

        # Channel-projection adapters -- ALWAYS trainable, independent of
        # `freeze`. New decoder-adjacent layers, not part of "Merlin
        # finetuning" (see training/trainer.py::build_optimizer's param-group
        # split).
        self.proj_e0 = nn.Conv3d(self.TAP_CHANNELS["e0"], out_channels[0], kernel_size=1)
        self.proj_e1 = nn.Conv3d(self.TAP_CHANNELS["e1"], out_channels[1], kernel_size=1)
        self.proj_e2 = nn.Conv3d(self.TAP_CHANNELS["e2"], out_channels[2], kernel_size=1)
        self.proj_e3 = nn.Conv3d(self.TAP_CHANNELS["e3"], out_channels[3], kernel_size=1)
        self.proj_e4 = nn.Conv3d(self.TAP_CHANNELS["e4"], out_channels[4], kernel_size=1)

    def _unfreeze_last_n_stages(self, n: int) -> None:
        """Coarse, stage-level mirror of TextEncoder._unfreeze_last_n_layers.

        ResNet152 has no flat `layers` ModuleList to slice the way a
        transformer does -- layer1..4 are Sequentials of [3,8,36,3]
        Bottleneck3d blocks respectively. n=1 unfreezes layer4 only, n=2
        unfreezes layer3+layer4, etc. The stem (conv1/bn1) is never unfrozen
        by this flag; only reachable via freeze=False (unfreezes everything,
        stem included).
        """
        stages = [self.layer4, self.layer3, self.layer2, self.layer1]
        for stage in stages[:n]:
            for p in stage.parameters():
                p.requires_grad_(True)

    @staticmethod
    def _to_hwd(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 4, 2).contiguous()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """image: (B, 1, 224, 224, 160) -- MerlinGridDataset's fixed grid.
        Returns (e0, e1, e2, e3, e4), channel-projected, in (H,W,D) layout
        matching `image`'s own axis order."""
        x = image.permute(0, 1, 4, 2, 3).contiguous()  # -> (B,1,160,224,224) (D,H,W), mirrors I3ResNet.forward
        x = torch.cat([x, x, x], dim=1)                 # -> (B,3,160,224,224), mimics RGB triplication

        x = self.relu(self.bn1(self.conv1(x)))           # (B,64,160,112,112)
        t0 = self._to_hwd(x)
        x = self.maxpool(x)                               # (B,64,80,56,56)
        x = self.layer1(x); t1 = self._to_hwd(x)          # (B,256,80,56,56)
        x = self.layer2(x); t2 = self._to_hwd(x)          # (B,512,40,28,28)
        x = self.layer3(x); t3 = self._to_hwd(x)          # (B,1024,20,14,14)
        x = self.layer4(x); t4 = self._to_hwd(x)          # (B,2048,10,7,7)

        return (
            self.proj_e0(t0), self.proj_e1(t1), self.proj_e2(t2),
            self.proj_e3(t3), self.proj_e4(t4),
        )
