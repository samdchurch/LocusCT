"""
Dataset variant for the Merlin-encoder Grounder model (model.encoder_type ==
"merlin"). Unlike GrounderDataset, which assumes pre-resampled-on-disk images
at a consistent XY size (spatial_mode="fixed") or resizes to a configurable
spatial_size, this preprocesses through Merlin's own fixed monai pipeline
(RAS orientation, 1.5x1.5x3mm spacing, HU clip [-1000,1000]->[0,1],
center-crop/pad to 224x224x160 -- mirrors merlin.data.monai_transforms.
ImageTransforms, see build_merlin_grid_transform), extended to a 2-key
(image, mask) transform so both land on IDENTICAL voxel grids (mask via
nearest-neighbor, excluded from intensity scaling).

Every sample lands at the exact same (1,224,224,160) shape by construction
(SpatialPadd+CenterSpatialCropd guarantee this), so pad_amounts is always
zero -- Trainer._trim_padding (training/trainer.py) is then a safe no-op,
same batch dict contract as GrounderDataset otherwise (image, mask, id,
pad_amounts, text keys), so Trainer can consume this dataset unchanged.

Do not mix this with GrounderDataset's own manifests/dirs unless their
"image"/"mask" values resolve to the same underlying NIfTI files -- there is
no cross-resampling/registration between this grid and GrounderDataset's own
(see the project plan's resolution/grid design decision).
"""
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from data.dataset import ResumableDistributedSampler, ResumableSampler
from models.merlin_encoder import MerlinEncoder

logger = logging.getLogger(__name__)


def build_merlin_grid_transform() -> Compose:
    """Mirrors merlin.data.monai_transforms.ImageTransforms exactly, extended
    to a 2-key (image, mask) Compose so image and mask are reoriented/
    resampled/cropped onto IDENTICAL voxel grids. mask uses nearest-neighbor
    resampling (preserves exact 0/1 label values) and is excluded from
    ScaleIntensityRanged (already binary, not an HU volume)."""
    keys = ["image", "mask"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 3), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=keys, spatial_size=list(MerlinEncoder.FULL_GRID_HWD)),
        CenterSpatialCropd(keys=keys, roi_size=list(MerlinEncoder.FULL_GRID_HWD)),
        ToTensord(keys=keys),
    ])


class MerlinGridDataset(Dataset):
    """Same manifest schema and tokenizer/embedding_cache contract as
    GrounderDataset (data/dataset.py), but preprocessed through Merlin's own
    fixed grid instead of GrounderDataset's HU-window/spatial_mode logic."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer_name: str,
        max_text_len: int = 256,
        image_dir: str = "",
        mask_dir: str = "",
        embedding_cache: str = "",
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        with open(manifest_path) as f:
            data = json.load(f)
        self.image_dir = Path(image_dir) if image_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.samples = self._filter_missing(data)
        if max_samples is not None and max_samples < len(self.samples):
            # Same seeded-random-subset approach as GrounderDataset, for the
            # same reason: reproducible across runs sharing seed+max_samples.
            indices = sorted(random.Random(seed).sample(range(len(self.samples)), max_samples))
            self.samples = [self.samples[i] for i in indices]

        self.max_text_len = max_text_len
        self.transform = build_merlin_grid_transform()

        self._emb_cache: dict | None = None
        self._emb_index: dict[str, int] | None = None
        self._emb_feats = None
        self._emb_masks = None
        if embedding_cache:
            p = Path(embedding_cache)
            if p.is_dir():
                import numpy as np
                with open(p / "index.json") as f:
                    self._emb_index = json.load(f)
                self._emb_feats = np.load(str(p / "text_feats.npy"), mmap_mode="r")
                self._emb_masks = np.load(str(p / "text_padding_mask.npy"), mmap_mode="r")
            else:
                self._emb_cache = torch.load(embedding_cache, map_location="cpu")
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _filter_missing(self, samples: list[dict]) -> list[dict]:
        """Drop samples whose image or mask file doesn't exist on disk (same
        logic as GrounderDataset._filter_missing)."""
        kept = []
        for sample in samples:
            image_path = self.image_dir / sample["image"] if self.image_dir else Path(sample["image"])
            mask_path = self.mask_dir / sample["mask"] if self.mask_dir else Path(sample["mask"])
            if not image_path.exists() or not mask_path.exists():
                logger.warning(f"Skipping missing sample: {sample.get('mask', sample.get('image'))}")
                continue
            kept.append(sample)
        if len(kept) < len(samples):
            logger.warning(f"Dropped {len(samples) - len(kept)}/{len(samples)} samples with missing files")
        return kept

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        image_path = str(self.image_dir / sample["image"]) if self.image_dir else sample["image"]
        mask_path = str(self.mask_dir / sample["mask"]) if self.mask_dir else sample["mask"]
        try:
            out = self.transform({"image": image_path, "mask": mask_path})
        except Exception:
            logger.warning(f"Failed to load/transform sample {sample.get('mask', sample.get('image'))} "
                            f"(image={image_path}, mask={mask_path}); skipping to next sample", exc_info=True)
            return self.__getitem__((idx + 1) % len(self.samples))

        item: dict[str, Any] = {
            "image": out["image"].float(),                # (1, 224, 224, 160)
            # Re-binarize after nearest-neighbor resample -- defensive/no-op
            # in the common case (nearest never interpolates), cheap insurance
            # against any accidental float drift.
            "mask": (out["mask"] > 0.5).float(),           # (1, 224, 224, 160)
            "id": sample["mask"],
            # SpatialPadd+CenterSpatialCropd always produce the exact fixed
            # grid above -- unlike GrounderDataset's per-sample variable
            # padding, there's never anything to trim.
            "pad_amounts": torch.zeros(3, dtype=torch.long),
        }

        if self._emb_index is not None:
            emb_idx = self._emb_index[sample["mask"]]
            item["text_feats"] = torch.from_numpy(self._emb_feats[emb_idx].copy())
            item["text_padding_mask"] = torch.from_numpy(self._emb_masks[emb_idx].copy())
        elif self._emb_cache is not None:
            emb = self._emb_cache[sample["mask"]]
            item["text_feats"] = emb["text_feats"]
            item["text_padding_mask"] = emb["text_padding_mask"]
        else:
            encoding = self.tokenizer(
                sample["sentence"],
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = encoding["input_ids"].squeeze(0)
            item["attention_mask"] = encoding["attention_mask"].squeeze(0)

        return item


def build_merlin_dataloader(
    manifest_path: str | list[str],
    cfg: dict,
    split: str,
    num_workers: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    """Mirrors data.dataset.build_dataloader's signature/sampler logic, but
    builds MerlinGridDataset instead and reads cfg["data"]["merlin"] (falling
    back to the top-level cfg["data"] keys when a data.merlin.* key isn't
    set, keeping the common single-dataset case config-light). No
    group_by_image support (out of scope -- see project plan)."""
    from torch.utils.data import ConcatDataset

    merlin_cfg = cfg["data"].get("merlin", {})
    entries = [manifest_path] if isinstance(manifest_path, str) else manifest_path

    default_image_dir = merlin_cfg.get("image_dir") or cfg["data"].get("image_dir", "")
    default_mask_dir = merlin_cfg.get("mask_dir") or cfg["data"].get("mask_dir", "")
    embedding_cache = merlin_cfg.get("embedding_cache") or cfg["data"].get("embedding_cache", "")

    def _make_dataset(entry: str | dict) -> MerlinGridDataset:
        if isinstance(entry, dict):
            path = entry["manifest"]
            image_dir = entry.get("image_dir", default_image_dir)
            mask_dir = entry.get("mask_dir", default_mask_dir)
        else:
            path = entry
            image_dir = default_image_dir
            mask_dir = default_mask_dir

        return MerlinGridDataset(
            manifest_path=path,
            tokenizer_name=cfg["model"]["text_encoder_name"],
            max_text_len=cfg["data"]["max_text_len"],
            image_dir=image_dir,
            mask_dir=mask_dir,
            embedding_cache=embedding_cache,
            max_samples=cfg["data"].get("max_samples"),
            seed=seed,
        )

    dataset = _make_dataset(entries[0]) if len(entries) == 1 else ConcatDataset([_make_dataset(e) for e in entries])
    nw = num_workers if num_workers is not None else cfg["data"]["num_workers"]
    is_train = (split == "train")

    if world_size > 1:
        if is_train:
            sampler = ResumableDistributedSampler(
                dataset, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True, seed=seed,
            )
        else:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        shuffle = False
        drop_last = False
    else:
        sampler = ResumableSampler(len(dataset), seed) if is_train else None
        shuffle = False
        drop_last = is_train

    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=drop_last,
        persistent_workers=nw > 0,
        prefetch_factor=cfg["data"].get("prefetch_factor", 4) if nw > 0 else None,
    )
