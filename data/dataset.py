import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# (lo, hi) HU window ranges for lung, soft-tissue, and bone channels
MULTI_WINDOWS: list[tuple[float, float]] = [(-1000.0, 300.0), (-150.0, 300.0), (-300.0, 1500.0)]


def load_nifti_canonical(path: str) -> np.ndarray:
    """
    Load a NIfTI file, reorient to RAS+, return array as (D, H, W).

    Shared with evaluate_rexgroundingct_val.py and visualize_rexgroundingct_eval.py so
    GT/pred masks and images are compared and displayed in exactly the space the model
    trains and infers in, rather than each file's raw on-disk orientation.
    """
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    arr = img.get_fdata(dtype=np.float32)
    # nibabel returns (W, H, D) in RAS; transpose to (D, H, W)
    arr = arr.transpose(2, 1, 0)
    return _normalize_depth_axis(arr, path)


def _normalize_depth_axis(arr: np.ndarray, path: str) -> np.ndarray:
    """
    Two axes are normally an equal-sized in-plane pair, with the
    through-plane (depth) axis differing in size. Some scans land that
    differing axis in a different position after RAS reorientation; roll
    it to axis 0 so every sample shares a consistent (D, H, W) layout
    and can be batched together.
    """
    s0, s1, s2 = arr.shape
    if s1 == s2:
        pass
    elif s0 == s2:
        arr = np.moveaxis(arr, 1, 0)
    elif s0 == s1:
        arr = np.moveaxis(arr, 2, 0)
    else:
        logger.warning(f"Could not determine a consistent depth axis for {path} with shape {arr.shape}; leaving as-is")
    return np.ascontiguousarray(arr)


class GrounderDataset(Dataset):
    """
    Loads NIfTI CT volumes, referring expressions, and binary segmentation masks.

    Manifest JSON format:
        [{"image": "path.nii.gz", "mask": "path.nii.gz", "sentence": "...",
          "region": "...", "finding": "..."}, ...]
    The "mask" path is used as the sample id (for embedding-cache keys and
    per-sample eval output).

    spatial_mode="fixed":
        Volumes are assumed to already be at a consistent XY size (e.g. 352×352).
        The Z (depth) dimension is padded to the nearest multiple of 16 so the
        UNet's 4-stage downsampling always produces integer spatial dims.
        pad_amounts[2] records how many Z slices were added.

    spatial_mode="resize":
        Volumes are resampled to spatial_size=(D, H, W) using trilinear interpolation
        for images and nearest-neighbor for masks.
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer_name: str,
        spatial_size: tuple[int, int, int],
        max_text_len: int = 256,
        hu_min: float = -1000.0,
        hu_max: float = 1000.0,
        multi_window: bool = False,
        spatial_mode: str = "fixed",
        augment: bool = False,
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
            # A random (not first-N) subset -- the manifest's own order may cluster by
            # source dataset or acquisition, which would bias a first-N slice. Sampled
            # via a seeded RNG (independent of torch/numpy's global RNG state, since
            # ResumableSampler's own reseeding elsewhere shouldn't perturb this) and
            # re-sorted by index so the selected subset stays reproducible across runs
            # sharing the same seed+max_samples -- e.g. every ablation study run against
            # the same seed sees the identical subset, making comparisons fair.
            indices = sorted(random.Random(seed).sample(range(len(self.samples)), max_samples))
            self.samples = [self.samples[i] for i in indices]
        self.spatial_size = spatial_size
        self.max_text_len = max_text_len
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.multi_window = multi_window
        self.spatial_mode = spatial_mode
        self.augment = augment

        self._emb_cache: dict | None = None
        self._emb_index: dict[str, int] | None = None
        self._emb_feats = None
        self._emb_masks = None
        if embedding_cache:
            p = Path(embedding_cache)
            if p.is_dir():
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
        """Drop samples whose image or mask file doesn't exist on disk."""
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

        image_path = self.image_dir / sample["image"] if self.image_dir else sample["image"]
        mask_path = self.mask_dir / sample["mask"] if self.mask_dir else sample["mask"]
        try:
            image_np = load_nifti_canonical(image_path)
            mask_np = load_nifti_canonical(mask_path)
        except Exception:
            logger.warning(f"Failed to load NIfTI for sample {sample.get('mask', sample.get('image'))} "
                            f"(image={image_path}, mask={mask_path}); skipping to next sample", exc_info=True)
            return self.__getitem__((idx + 1) % len(self.samples))

        image_np = self._apply_windows(image_np)  # always (C, D, H, W)

        if self.spatial_mode == "fixed":
            image, mask, pad_amounts = self._pad_to_divisible(image_np, mask_np)
        else:
            image = self._resize_volume(image_np, self.spatial_size, is_mask=False)
            mask = self._resize_volume(mask_np, self.spatial_size, is_mask=True)
            pad_amounts = torch.zeros(3, dtype=torch.long)

        if self.augment:
            image, mask = self._augment(image, mask)

        item: dict[str, Any] = {
            "image": image,        # (C, D, H, W) float32  C=1 or 3
            "mask": mask.float(),  # (1, D, H, W) float32
            "id": sample["mask"],
            "pad_amounts": pad_amounts,  # (3,) int64
        }

        if self._emb_index is not None:
            idx = self._emb_index[sample["mask"]]
            item["text_feats"] = torch.from_numpy(self._emb_feats[idx].copy())
            item["text_padding_mask"] = torch.from_numpy(self._emb_masks[idx].copy())
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
            item["input_ids"] = encoding["input_ids"].squeeze(0)           # (L,) int64
            item["attention_mask"] = encoding["attention_mask"].squeeze(0) # (L,) int64

        return item

    def _apply_windows(self, vol: np.ndarray) -> np.ndarray:
        """Normalize HU values; returns (C, D, H, W) where C=1 or C=3 (multi_window)."""
        if self.multi_window:
            channels = []
            for lo, hi in MULTI_WINDOWS:
                c = np.clip(vol, lo, hi)
                c = (c - lo) / (hi - lo) * 2.0 - 1.0
                channels.append(c)
            return np.stack(channels, axis=0)  # (3, D, H, W)
        vol = np.clip(vol, self.hu_min, self.hu_max)
        vol = (vol - self.hu_min) / (self.hu_max - self.hu_min) * 2.0 - 1.0
        return vol[np.newaxis]  # (1, D, H, W)

    def _resize_volume(
        self,
        vol: np.ndarray,
        target: tuple[int, int, int],
        is_mask: bool = False,
    ) -> torch.Tensor:
        """Resize volume to target.

        Mask input is (D, H, W) → returns (1, D, H, W).
        Image input is (C, D, H, W) → returns (C, D, H, W).
        """
        t = torch.from_numpy(vol).float()
        if is_mask:
            t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        else:
            t = t.unsqueeze(0)  # (1, C, D, H, W)
        mode = "nearest" if is_mask else "trilinear"
        align = None if is_mask else False
        t = F.interpolate(t, size=target, mode=mode, align_corners=align)
        return t.squeeze(0)  # (1, D, H, W) for mask, (C, D, H, W) for image

    def _pad_to_divisible(
        self,
        image_np: np.ndarray,
        mask_np: np.ndarray,
        divisor: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad each spatial dim to the nearest multiple of `divisor`.
        Padding is applied symmetrically (extra on the end).
        Returns image (C,D,H,W), mask (1,D,H,W), pad_amounts (3,).
        """
        D, H, W = image_np.shape[-3:]  # works for (C, D, H, W)
        pad_D = (divisor - D % divisor) % divisor
        pad_H = (divisor - H % divisor) % divisor
        pad_W = (divisor - W % divisor) % divisor

        image = torch.from_numpy(image_np)             # (C, D, H, W)
        mask = torch.from_numpy(mask_np).unsqueeze(0)  # (1, D, H, W)

        # F.pad order: last dim first → (W_before, W_after, H_before, H_after, D_before, D_after)
        padding = (0, pad_W, 0, pad_H, 0, pad_D)
        image = F.pad(image.float(), padding, mode="constant", value=-1.0)
        mask = F.pad(mask.float(), padding, mode="constant", value=0.0)

        pad_amounts = torch.tensor([pad_D, pad_H, pad_W], dtype=torch.long)
        return image, mask, pad_amounts

    def _augment(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation: intensity shift and scale only (no spatial flips)."""
        # Additive intensity noise ±0.1
        shift = (torch.rand(1).item() * 2 - 1) * 0.1
        image = image + shift
        # Multiplicative scale ±10%
        scale = 1.0 + (torch.rand(1).item() * 2 - 1) * 0.1
        image = image * scale
        image = image.clamp(-1.0, 1.0)
        return image, mask


class GrounderImageGroupedDataset(GrounderDataset):
    """
    Same manifest/config surface as GrounderDataset, but indexed by unique
    image rather than by (image, sentence, mask) triplet: one __getitem__
    call loads the image ONCE and returns its (sentence, mask) findings
    stacked as a batch of N, for Grounder's shared-image-encoding training
    mode (see UNet3D.forward's image-batch-1/text-batch-N broadcast).
    Opt-in via data.group_by_image=true, train split only -- see
    build_dataloader.

    max_findings_per_image: if an image has more findings than this, a
    random subset of that many is used for a given __getitem__ call instead
    of all of them (re-sampled every call, so different subsets get seen
    across epochs) -- caps per-step decoder batch size/memory for images
    with unusually many findings. None (default): no cap, use every finding
    every time, matching the original all-at-once behavior.
    """

    def __init__(self, *args: Any, max_findings_per_image: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_findings_per_image = max_findings_per_image
        groups: dict[str, list[dict]] = {}
        for s in self.samples:
            groups.setdefault(s["image"], []).append(s)
        self.image_groups: list[list[dict]] = list(groups.values())

    def __len__(self) -> int:
        return len(self.image_groups)

    def _pad_group_to_divisible(
        self,
        image_np: np.ndarray,
        mask_arrs: list[np.ndarray],
        divisor: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same pad-amount math as GrounderDataset._pad_to_divisible, applied
        to the image once and to the whole (N, D, H, W) mask stack at once."""
        D, H, W = image_np.shape[-3:]
        pad_D = (divisor - D % divisor) % divisor
        pad_H = (divisor - H % divisor) % divisor
        pad_W = (divisor - W % divisor) % divisor

        image = torch.from_numpy(image_np)                                    # (C, D, H, W)
        mask_stack = torch.from_numpy(np.stack(mask_arrs, axis=0)).unsqueeze(1)  # (N, 1, D, H, W)

        padding = (0, pad_W, 0, pad_H, 0, pad_D)
        image = F.pad(image.float(), padding, mode="constant", value=-1.0)
        mask_stack = F.pad(mask_stack.float(), padding, mode="constant", value=0.0)

        pad_amounts = torch.tensor([pad_D, pad_H, pad_W], dtype=torch.long)
        return image, mask_stack, pad_amounts

    def _resize_mask_stack(self, mask_arrs: list[np.ndarray], target: tuple[int, int, int]) -> torch.Tensor:
        """F.interpolate already handles an arbitrary leading batch dim, so the
        whole (N, D, H, W) stack is resized in one call, no per-mask loop."""
        t = torch.from_numpy(np.stack(mask_arrs, axis=0)).float().unsqueeze(1)  # (N, 1, D, H, W)
        return F.interpolate(t, size=target, mode="nearest")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        group = self.image_groups[idx]
        if self.max_findings_per_image is not None and len(group) > self.max_findings_per_image:
            # Sampled fresh (not cached on self.image_groups) so a different
            # subset of this image's findings gets used across epochs, before
            # any per-mask load/shape checks -- caps I/O too, not just the
            # eventual decoder batch size.
            group = random.sample(group, self.max_findings_per_image)
        image_path = self.image_dir / group[0]["image"] if self.image_dir else group[0]["image"]
        try:
            image_np = load_nifti_canonical(image_path)
        except Exception:
            logger.warning(f"Failed to load image {image_path}; skipping group", exc_info=True)
            return self.__getitem__((idx + 1) % len(self.image_groups))
        image_np = self._apply_windows(image_np)  # (C, D, H, W)

        image_shape = image_np.shape[-3:]
        kept: list[dict] = []
        mask_arrs: list[np.ndarray] = []
        for s in group:
            mask_path = self.mask_dir / s["mask"] if self.mask_dir else s["mask"]
            try:
                mask_np = load_nifti_canonical(mask_path)
            except Exception:
                logger.warning(f"Failed to load mask {mask_path}; dropping this finding", exc_info=True)
                continue
            if mask_np.shape != image_shape:
                # Known data issue (see evaluate_finetuned_voxtell_ed.py) --
                # the mask was likely drawn on a different reconstruction/
                # series than "image" points to. Padding a mismatched mask
                # against pad_amounts computed from the image's own shape
                # would silently produce a wrong (or, if the image happens
                # to already be divisor-aligned, unpadded and mismatched)
                # tensor instead of a clean error -- drop it like an
                # unreadable file rather than stacking it into the group.
                logger.warning(
                    f"Mask {mask_path} shape {mask_np.shape} != image {image_path} shape "
                    f"{image_shape}; dropping this finding"
                )
                continue
            mask_arrs.append(mask_np)
            kept.append(s)
        if not kept:
            return self.__getitem__((idx + 1) % len(self.image_groups))

        if self.spatial_mode == "fixed":
            image, mask_stack, pad_amounts = self._pad_group_to_divisible(image_np, mask_arrs)
        else:
            image = self._resize_volume(image_np, self.spatial_size, is_mask=False)
            mask_stack = self._resize_mask_stack(mask_arrs, self.spatial_size)
            pad_amounts = torch.zeros(3, dtype=torch.long)

        if self.augment:
            image, _ = self._augment(image, mask_stack)

        item: dict[str, Any] = {
            "image": image.unsqueeze(0),       # (1, C, D, H, W)
            "mask": mask_stack.float(),         # (N, 1, D, H, W)
            "id": [s["mask"] for s in kept],    # list[str], len N
            "pad_amounts": pad_amounts,         # (3,) -- shared, image-derived
        }

        if self._emb_index is not None:
            feats = np.stack([self._emb_feats[self._emb_index[s["mask"]]] for s in kept], axis=0)
            masks = np.stack([self._emb_masks[self._emb_index[s["mask"]]] for s in kept], axis=0)
            item["text_feats"] = torch.from_numpy(feats.copy())
            item["text_padding_mask"] = torch.from_numpy(masks.copy())
        elif self._emb_cache is not None:
            item["text_feats"] = torch.stack([self._emb_cache[s["mask"]]["text_feats"] for s in kept])
            item["text_padding_mask"] = torch.stack([self._emb_cache[s["mask"]]["text_padding_mask"] for s in kept])
        else:
            encoding = self.tokenizer(
                [s["sentence"] for s in kept],
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item["input_ids"] = encoding["input_ids"]            # (N, L) int64
            item["attention_mask"] = encoding["attention_mask"]  # (N, L) int64

        return item


class ResumableSampler(Sampler):
    """
    Single-process per-epoch shuffle, reseeded as (seed + epoch) so the order
    is a pure function of epoch number -- reproducible across process
    restarts, unlike a single persistent generator advancing across epochs
    (which regenerates epoch 0's order after a restart, not epoch N's).

    set_epoch(epoch, skip_samples) supports resuming mid-epoch: skip_samples
    drops that many indices off the front of this epoch's already-determined
    shuffle order, so a resumed run sees only the samples it hasn't yet
    processed, in the same order it would have -- not a fresh shuffle.
    """

    def __init__(self, dataset_len: int, seed: int) -> None:
        self.dataset_len = dataset_len
        self.seed = seed
        self.epoch = 0
        self.skip_samples = 0

    def set_epoch(self, epoch: int, skip_samples: int = 0) -> None:
        self.epoch = epoch
        self.skip_samples = skip_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_len, generator=g).tolist()
        return iter(indices[self.skip_samples:])

    def __len__(self) -> int:
        return self.dataset_len - self.skip_samples


class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler whose set_epoch also accepts skip_samples, applied
    after this rank's normal per-epoch index order is computed -- see
    ResumableSampler's docstring for why this is needed for mid-epoch resume."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.skip_samples = 0

    def set_epoch(self, epoch: int, skip_samples: int = 0) -> None:
        super().set_epoch(epoch)
        self.skip_samples = skip_samples

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.skip_samples:])

    def __len__(self) -> int:
        return super().__len__() - self.skip_samples


def build_dataloader(
    manifest_path: str | list[str],
    cfg: dict,
    split: str,
    num_workers: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    from torch.utils.data import ConcatDataset

    augment = (split == "train") and cfg["data"].get("augment_train", True)
    entries = [manifest_path] if isinstance(manifest_path, str) else manifest_path

    default_image_dir = cfg["data"].get("image_dir", "")
    default_mask_dir = cfg["data"].get("mask_dir", "")

    # Train split only: sample one image, run the encoder once, and train on
    # ALL of its (phrase, mask) findings in one step (see
    # GrounderImageGroupedDataset / UNet3D.forward's shared-encoder broadcast).
    # Opt-in so every existing config keeps today's per-triplet behavior.
    group_by_image = (split == "train") and cfg["data"].get("group_by_image", False)
    dataset_cls = GrounderImageGroupedDataset if group_by_image else GrounderDataset

    def _make_dataset(entry: str | dict) -> GrounderDataset:
        # A manifest entry is normally just a path, resolved against the shared
        # image_dir/mask_dir. A dict lets one manifest (e.g. one drawing on a
        # differently-rooted dataset like ReXGroundingCT) override those roots
        # without moving every other manifest's paths onto a shared root too.
        if isinstance(entry, dict):
            path = entry["manifest"]
            image_dir = entry.get("image_dir", default_image_dir)
            mask_dir = entry.get("mask_dir", default_mask_dir)
        else:
            path = entry
            image_dir = default_image_dir
            mask_dir = default_mask_dir

        # Only GrounderImageGroupedDataset accepts this -- GrounderDataset has
        # no per-image concept of "findings" to cap.
        extra_kwargs = {}
        if group_by_image:
            extra_kwargs["max_findings_per_image"] = cfg["data"].get("max_findings_per_image")

        return dataset_cls(
            manifest_path=path,
            tokenizer_name=cfg["model"]["text_encoder_name"],
            spatial_size=tuple(cfg["data"]["spatial_size"]),
            max_text_len=cfg["data"]["max_text_len"],
            hu_min=cfg["data"]["hu_min"],
            hu_max=cfg["data"]["hu_max"],
            multi_window=cfg["data"].get("multi_window", False),
            spatial_mode=cfg["data"]["spatial_mode"],
            augment=augment,
            image_dir=image_dir,
            mask_dir=mask_dir,
            embedding_cache=cfg["data"].get("embedding_cache", ""),
            max_samples=cfg["data"].get("max_samples"),
            seed=seed,
            **extra_kwargs,
        )

    dataset = _make_dataset(entries[0]) if len(entries) == 1 else ConcatDataset([_make_dataset(e) for e in entries])
    nw = num_workers if num_workers is not None else cfg["data"]["num_workers"]
    is_train = (split == "train")

    if world_size > 1:
        if is_train:
            # Resumable variant: set_epoch(epoch, skip_samples) lets the trainer
            # resume mid-epoch without re-consuming already-seen samples.
            sampler = ResumableDistributedSampler(
                dataset, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True, seed=seed,
            )
        else:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        shuffle = False
        drop_last = False
    else:
        # ResumableSampler reseeds as (seed + epoch) itself -- unlike a single
        # persistent generator advancing across epochs, this makes shuffle
        # order a pure function of epoch number, reproducible across process
        # restarts, and lets the trainer resume mid-epoch via set_epoch's
        # skip_samples. No sampler needed for val (never shuffled).
        sampler = ResumableSampler(len(dataset), seed) if is_train else None
        shuffle = False
        drop_last = is_train

    return DataLoader(
        dataset,
        # batch_size=None disables DataLoader's own collation, yielding each
        # __getitem__ result as-is -- GrounderImageGroupedDataset already
        # returns one fully-formed training step (image + all its findings).
        batch_size=None if group_by_image else cfg["training"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=cfg["data"].get("pin_memory", True),
        # DataLoader raises if batch_size=None and drop_last is truthy (an
        # explicit incompatibility, not just inert as assumed) -- force it
        # off in that mode rather than relying on the branches above.
        drop_last=False if group_by_image else drop_last,
        persistent_workers=nw > 0,
        prefetch_factor=cfg["data"].get("prefetch_factor", 4) if nw > 0 else None,
    )
