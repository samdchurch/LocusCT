#!/usr/bin/env python3
"""
Fine-tune the full text-grounded VoxTell model (encoder + Qwen3 text-fusion
decoder) on our own referring-expression dataset, with the SAME single
raw-HU input channel and preprocessing the original paper/pretrained
checkpoint uses -- see "Single-channel input" below. Pass --multi-window to
opt into our own experimental 3-channel (lung/soft-tissue/bone) windowed
input instead.

This is NOT the fine-tuning path VoxTell's own repo ships (`voxtell-finetune`,
see voxtell/training/voxtell_trainer.py): that one transfers only the
pretrained image encoder into a fresh, closed-set multi-class nnU-Net decoder,
discarding the free-text prompting entirely. This script instead continues
training the *whole* VoxTellModel end-to-end (voxtell/model/voxtell_model.py),
so it keeps open-vocabulary referring expressions like Grounder's own model --
there is no official reference implementation for this, so the training loop
below (loss, schedule, patch sampling) is our own design, not VoxTell's.

Single-channel input (default): matches voxtell/voxtell_v1.1/plans.json's
own "normalization_schemes": ["ZScoreNormalization"] -- whole-volume
(x - mean) / std, no HU clipping, same as nnU-Net's un-clipped per-case
ZScoreNormalization (not "CTNormalization", which nnU-Net reserves for
global-dataset-statistics HU clipping and isn't what this checkpoint's
plans.json specifies). No stem-conv channel adaptation is needed since the
input channel count already matches the pretrained checkpoint.

Multi-window input (--multi-window, opt-in): our own experimental 3-channel
(lung/soft-tissue/bone) windowed input, NOT the original paper's scheme.
VoxTell's pretrained stem conv only has 1 input channel, so the stem conv
weight is repeated/averaged 1 -> 3 channels before loading (the same trick
voxtell_trainer.py's _load_encoder_weights uses for its own encoder-only
transfer) -- everything else loads unchanged. Window ranges are
data.dataset.MULTI_WINDOWS, the same convention configs/default.yaml's
data.multi_window flag already uses for our own model, clipped and rescaled
to [-1, 1] per channel.

Image resolution: pretrained-checkpoint fine-tuning (default) keeps
--image-dir/--mask-dir on the NATIVE (pre-resample) directories, since
VoxTell does not resample images to a standard spacing itself (see
evaluate_voxtell_ed.py / evaluate_voxtell_rexgroundingct.py's own
docstrings) and the pretrained checkpoint degrades on spacings it wasn't
trained on -- moving it to Grounder's resampled grid would depart from that
pretraining distribution for reasons unrelated to any real model-quality
comparison. --from-scratch has no such pretrained prior to protect, so it
defaults --image-dir/--mask-dir to configs/default.yaml's own
nifti_resampled/labels_resampled instead, matching Grounder's exact input
grid for a fair from-scratch-vs-from-scratch comparison. Either way,
resample_and_crop.py/resample_masks.py preserve each manifest entry's
relative path, so the same manifest JSON files Grounder trains on are reused
unchanged, just resolved against a different base directory -- and either
way VoxTellModel still only ever sees a 192^3 patch cropped/tiled out of
whichever volume gets loaded (see "Patch size is fixed at 192^3" below); the
resampled grid is never fed to it in one shot.

Patch size is fixed at 192^3: VoxTellModel precomputes a sinusoidal
positional-encoding buffer sized for a 192^3 input patch (see
DECODER_CONFIGS in voxtell_model.py) and isn't parameterized to recompute it
for another patch size.

Random crops (--random-crop-fraction, train split only): by default, 33% of
training patches come from random_patch (a uniformly random 192^3 crop, no
guarantee it touches the mask's foreground at all) instead of
foreground_patch (guaranteed to contain the whole foreground). Real
sliding-window eval (evaluate_finetuned_voxtell_ed.py) hands the model
plenty of empty/foreground-clipped tiles; foreground_patch alone never
trains on one, so the model never learns to predict "no match" for its own
text prompt or to segment a mask cut off at a patch boundary. Val/held-out
splits are never affected by this flag -- they stay 100% foreground_patch/
deterministic so Macro Hit Rate/early stopping keep a stable signal across
epochs. --random-crop-fraction 0.0 restores the old (pre-this-flag) 100%
foreground_patch behavior for train too.

Validation Dice here is a fast proxy (one foreground-centered 192^3 patch per
sample, not full-volume sliding-window inference) -- use evaluate_voxtell_
ed.py / evaluate_voxtell_rexgroundingct.py-style sliding-window evaluation
for a real held-out number.

Macro Hit Rate / early stopping: same definition and mechanism as training/
trainer.py's Trainer.macro_hit_rate_epoch/early_stop_patience (ED_CATEGORIES/
FINDING_TO_ED_CATEGORY imported from there directly, not duplicated) -- the
mean of ONC's plain hit rate (dice >= 0.1) and the average of ED's 13
per-category hit rates, evaluated each epoch against --ed-val-manifest/
--onc-val-manifest (same fast single-patch proxy as Validation Dice above,
not sliding-window). Training stops once it hasn't improved for
--early-stop-patience epochs. Pass --ed-val-manifest ""/--onc-val-manifest ""
to disable (then --num-epochs is the only stopping condition, as before).

Text embedding cache: the text backbone is frozen (embed_sentences is
@torch.no_grad()), so a given sentence produces the same embedding every
epoch -- re-encoding it live every batch is redundant, and under DDP every
rank separately loads its own copy of the 4B-parameter backbone for no
benefit. Pass --embedding-cache <dir> (built by
scripts/embedding/precompute_voxtell_embeddings.py) to load precomputed
embeddings instead -- the text backbone is then never loaded at all. Same
optional convention as Grounder's own data.embedding_cache; default is live
encoding.

Training from scratch (--from-scratch): keeps VoxTellModel's architecture
(same arch_kwargs/decoder_layer/text_embedding_dim/etc. from plans.json, see
build_voxtell_model) but skips loading --model-dir's pretrained
checkpoint_final.pth, so the image encoder/decoder start from VoxTellModel's
own random init. The text encoder is still frozen either way (this script
never trains it). Unless explicitly overridden, --lr/--weight-decay/
--warmup-epochs/--multi-window switch to configs/default.yaml's own
from-scratch Grounder training defaults (2e-4/1e-5/0/on) instead of this
script's finetuning-tuned defaults (1e-5/1e-5/5/off); --image-dir/--mask-dir
switch to configs/default.yaml's resampled nifti_resampled/labels_resampled
grid instead of this script's native-resolution finetuning defaults, per
"Image resolution" above.

Usage
-----
    python finetune_voxtell.py
    python finetune_voxtell.py --batch-size 2 --num-epochs 50 --lr 1e-4
    python finetune_voxtell.py --resume runs/voxtell_finetune/checkpoints/epoch_0010.pt
    python finetune_voxtell.py --embedding-cache /path/to/voxtell_embeddings_mmap
    python finetune_voxtell.py --from-scratch --output-dir runs/voxtell_from_scratch

    # Multi-GPU (DDP, same pattern as train.py) -- see submit_finetune_voxtell.sh
    python -m torch.distributed.run --nproc_per_node=4 finetune_voxtell.py
"""

import argparse
import json
import logging
import math
import os
import pydoc
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction

from data.dataset import MULTI_WINDOWS, load_nifti_canonical
from training.losses import CombinedLoss
from training.trainer import ED_CATEGORIES, FINDING_TO_ED_CATEGORY
from utils.metrics import dice_score, iou_score

DEFAULT_TRAIN_MANIFEST = "official_splits/all_data_train.json"
DEFAULT_VAL_MANIFEST = "official_splits/curated_ed_onc_val_data.json"
DEFAULT_ED_VAL_MANIFEST = "official_splits/curated_ed_val_data.json"
DEFAULT_ONC_VAL_MANIFEST = "official_splits/curated_onc_val_data.json"
# Native (pre-resample) roots -- default for pretrained-checkpoint
# fine-tuning; see module docstring's "Image resolution" section.
DEFAULT_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti"
DEFAULT_MASK_DIR = "/path/to/data/inhouse_abdominal_ct/labels"
# configs/default.yaml's resampled grid -- default for --from-scratch instead.
DEFAULT_RESAMPLED_IMAGE_DIR = "/path/to/data/inhouse_abdominal_ct/nifti_resampled"
DEFAULT_RESAMPLED_MASK_DIR = "/path/to/data/inhouse_abdominal_ct/labels_resampled"
DEFAULT_MODEL_DIR = Path(__file__).parent / "voxtell" / "voxtell_v1.1"
DEFAULT_TEXT_ENCODER = "/path/to/data/models/Qwen3-Embedding-4B"

PATCH_SIZE = (192, 192, 192)  # default for the published 192^3 checkpoint; overridden via --patch-size per-run

# VoxTellModel registers the stem conv under both a top-level `encoder.` path
# and a `decoder.encoder.` path (same underlying weights, two attributes) --
# both must be adapted or load_state_dict only fixes one and size-mismatches
# on the other. `.conv.weight` vs `.all_modules.0.weight` is a separate
# aliasing (torch.compile wraps modules differently), same pair
# voxtell_trainer.py's _load_encoder_weights adapts.
STEM_CONV_KEYS = (
    "encoder.stem.convs.0.conv.weight",
    "encoder.stem.convs.0.all_modules.0.weight",
    "decoder.encoder.stem.convs.0.conv.weight",
    "decoder.encoder.stem.convs.0.all_modules.0.weight",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--ed-val-manifest", default=DEFAULT_ED_VAL_MANIFEST,
                         help="Held-out ED manifest for Macro Hit Rate/early stopping (see module docstring); "
                              "empty string disables Macro Hit Rate tracking and early stopping")
    parser.add_argument("--onc-val-manifest", default=DEFAULT_ONC_VAL_MANIFEST,
                         help="Held-out ONC manifest for Macro Hit Rate/early stopping; empty string disables")
    parser.add_argument("--early-stop-patience", type=int, default=4,
                         help="Stop once Macro Hit Rate hasn't improved for this many epochs; "
                              "0 disables early stopping (train.py/configs/default.yaml default is also 4)")
    parser.add_argument("--image-dir", default=None,
                         help=f"Default: native {DEFAULT_IMAGE_DIR}, or resampled "
                              f"{DEFAULT_RESAMPLED_IMAGE_DIR} with --from-scratch (see module docstring)")
    parser.add_argument("--mask-dir", default=None,
                         help=f"Default: native {DEFAULT_MASK_DIR}, or resampled "
                              f"{DEFAULT_RESAMPLED_MASK_DIR} with --from-scratch (see module docstring)")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                         help="VoxTell model directory (plans.json + fold_0/checkpoint_final.pth)")
    parser.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    parser.add_argument("--patch-size", type=int, default=192,
                         help="Cubic input patch side length (N,N,N). The published voxtell_v1.1 checkpoint "
                              "is hard-pinned to 192 -- only pass a different value together with "
                              "--from-scratch and a --model-dir designed for it (see "
                              "voxtell/voxtell_scratch_96/plans.json).")
    parser.add_argument("--num-maskformer-stages", type=int, default=5,
                         help="Passed to VoxTellModel -- see build_voxtell_model. Default (5) matches both "
                              "the published 192^3 checkpoint and voxtell_scratch_96.")
    parser.add_argument("--decoder-layer", type=int, default=4,
                         help="Passed to VoxTellModel -- see build_voxtell_model. Default (4) matches both "
                              "the published 192^3 checkpoint and voxtell_scratch_96.")
    parser.add_argument("--text-embedding-dim", type=int, default=2560,
                         help="Must equal --text-encoder's hidden_size (2560 for Qwen3-Embedding-4B, the "
                              "published checkpoint's own encoder and default here; 4096 for Qwen3-Embedding-8B). "
                              "Only meaningful to change together with --from-scratch -- the published checkpoint "
                              "is pinned to 2560.")
    parser.add_argument("--embedding-cache", default="",
                         help="Dir from precompute_voxtell_embeddings.py -- skips loading the live "
                              "text backbone entirely and loads cached embeddings instead "
                              "(default: empty, live encoding)")
    parser.add_argument("--multi-window", dest="multi_window", action="store_true", default=None,
                         help="Use our own experimental 3-window (lung/soft-tissue/bone) stack instead of "
                              "the original paper's single Z-score-normalized channel "
                              "(default: off, except --from-scratch defaults it on -- see --from-scratch)")
    parser.add_argument("--no-multi-window", dest="multi_window", action="store_false", default=None,
                         help="Force the original paper's single Z-score-normalized channel (input_channels=1) "
                              "even under --from-scratch, which otherwise defaults --multi-window on. No effect "
                              "without --from-scratch, since plain finetuning already defaults to single-channel.")
    parser.add_argument("--random-crop-fraction", type=float, default=0.33,
                         help="Train-split-only: fraction of samples that get a truly random crop "
                              "(random_patch) instead of a foreground-guaranteed one (foreground_patch) -- "
                              "may land empty or clip the mask at the patch boundary, matching what real "
                              "sliding-window eval hands the model. Val/held-out splits always stay "
                              "foreground-guaranteed/deterministic, unaffected by this flag. 0.0 disables "
                              "(100%% foreground_patch, pre-existing behavior). Default is more conservative "
                              "than nnU-Net's own oversample_foreground_percent=0.33 convention (which makes "
                              "*67%%* of crops random, not 33%%) given this task's single-small-lesion-per-"
                              "sample regime vs. nnU-Net's closed-set organ segmentation.")
    parser.add_argument("--from-scratch", action="store_true", default=False,
                         help="Randomly initialize VoxTell's image encoder/decoder instead of loading "
                              "--model-dir's pretrained checkpoint (text encoder is still frozen either way). "
                              "Unless overridden, also switches --lr/--weight-decay/--warmup-epochs/--multi-window "
                              "to configs/default.yaml's from-scratch Grounder training defaults "
                              "(2e-4/1e-5/0/on) instead of this script's finetuning defaults, and switches "
                              "--image-dir/--mask-dir to the resampled nifti_resampled/labels_resampled grid "
                              "instead of native resolution")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--constant-lr", action="store_true",
                         help="Hold --lr fixed for the whole run instead of the warmup+cosine schedule")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Truncate each split for a quick smoke-test run")
    parser.add_argument("--output-dir", default="runs/voxtell_finetune")
    parser.add_argument("--keep-last-n", type=int, default=1,
                         help="How many epoch_*.pt checkpoints to keep, most recent first -- best.pt is "
                              "separate and always kept regardless of this value.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--resume-partial", action="store_true",
        help="Warm-start model weights from --resume (strict=False) instead of resuming a run: "
             "optimizer/scheduler state, epoch, best_dice, and best_macro_hit_rate are NOT restored, "
             "so --lr/--num-epochs take effect and training starts at epoch 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.patch_size = (args.patch_size, args.patch_size, args.patch_size)

    # --from-scratch's hyperparameter defaults mirror configs/default.yaml's
    # optimizer/scheduler/data sections (Grounder's own from-scratch training
    # config), overriding this script's finetuning-tuned defaults -- but only
    # for whichever of --lr/--weight-decay/--warmup-epochs/--multi-window/
    # --image-dir/--mask-dir the caller didn't already pass explicitly.
    if args.from_scratch:
        if args.lr is None:
            args.lr = 2.0e-4
        if args.weight_decay is None:
            args.weight_decay = 1.0e-5
        if args.warmup_epochs is None:
            args.warmup_epochs = 0
        if args.multi_window is None:
            args.multi_window = True
        if args.image_dir is None:
            args.image_dir = DEFAULT_RESAMPLED_IMAGE_DIR
        if args.mask_dir is None:
            args.mask_dir = DEFAULT_RESAMPLED_MASK_DIR
    else:
        if args.lr is None:
            args.lr = 1e-5
        if args.weight_decay is None:
            args.weight_decay = 1e-5
        if args.warmup_epochs is None:
            args.warmup_epochs = 5
        if args.multi_window is None:
            args.multi_window = False
        if args.image_dir is None:
            args.image_dir = DEFAULT_IMAGE_DIR
        if args.mask_dir is None:
            args.mask_dir = DEFAULT_MASK_DIR
    return args


def apply_windows(vol: np.ndarray, multi_window: bool) -> np.ndarray:
    """
    multi_window=True: our own experimental 3-channel (lung/soft-tissue/bone)
    windowing, clipped and rescaled to [-1, 1] per channel -- see module
    docstring for why this is NOT the original paper's scheme.

    multi_window=False (default): single raw-HU channel, whole-volume
    Z-score normalized ((x - mean) / std, no clipping) -- matches
    voxtell/voxtell_v1.1/plans.json's own normalization_schemes
    (["ZScoreNormalization"]), i.e. the original paper/pretrained
    checkpoint's own preprocessing.
    """
    if multi_window:
        channels = []
        for lo, hi in MULTI_WINDOWS:
            c = np.clip(vol, lo, hi)
            c = (c - lo) / (hi - lo) * 2.0 - 1.0
            channels.append(c)
        return np.stack(channels, axis=0)  # (3, D, H, W)
    mean = vol.mean()
    std = vol.std()
    vol = (vol - mean) / max(std, 1e-8)
    return vol[np.newaxis].astype(np.float32)  # (1, D, H, W)


def _crop_and_pad(
    image: torch.Tensor,
    mask: torch.Tensor,
    start: list[int],
    patch_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop patch_size out of (image, mask) at `start`, padding (image with -1.0 /
    "air", matching GrounderDataset's own padding value; mask with 0.0) if the
    volume is smaller than patch_size in some dim. Shared by foreground_patch and
    random_patch, which differ only in how `start` is chosen."""
    d0, h0, w0 = start
    D, H, W = image.shape[1:]
    d1, h1, w1 = min(d0 + patch_size[0], D), min(h0 + patch_size[1], H), min(w0 + patch_size[2], W)

    image_patch = image[:, d0:d1, h0:h1, w0:w1]
    mask_patch = mask[d0:d1, h0:h1, w0:w1]

    pad = (0, patch_size[2] - (w1 - w0), 0, patch_size[1] - (h1 - h0), 0, patch_size[0] - (d1 - d0))
    if any(pad):
        image_patch = F.pad(image_patch, pad, mode="constant", value=-1.0)
        mask_patch = F.pad(mask_patch.unsqueeze(0), pad, mode="constant", value=0.0).squeeze(0)

    return image_patch, mask_patch.unsqueeze(0)  # (C,*patch), (1,*patch)


def foreground_patch(
    image: torch.Tensor,
    mask: torch.Tensor,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Crop (or pad) a patch_size patch containing the mask's foreground.

    image: (C, D, H, W) float32.  mask: (D, H, W) float32 (0/1).
    If the foreground bbox fits within patch_size in a given dim, the crop
    start is randomized (deterministic=False) within the range that keeps the
    whole bbox inside the patch; otherwise it's centered on the bbox. Volumes
    smaller than patch_size are padded (image with -1.0 / "air", matching
    GrounderDataset's own padding value; mask with 0.0).
    """
    vol_shape = image.shape[1:]
    fg = torch.nonzero(mask > 0.5, as_tuple=False)
    if fg.numel() == 0:
        bbox_min = bbox_max = [s // 2 for s in vol_shape]
    else:
        bbox_min = fg.min(dim=0).values.tolist()
        bbox_max = fg.max(dim=0).values.tolist()

    starts = []
    for dim, p in enumerate(patch_size):
        size = vol_shape[dim]
        if size <= p:
            starts.append(0)
            continue
        lo, hi = bbox_min[dim], bbox_max[dim]
        lowest_start = max(0, hi - p + 1)
        highest_start = min(size - p, lo)
        if lowest_start > highest_start:
            # bbox itself is larger than the patch in this dim -- center on it
            start = max(0, min(size - p, (lo + hi) // 2 - p // 2))
        elif deterministic:
            start = (lowest_start + highest_start) // 2
        else:
            start = random.randint(lowest_start, highest_start)
        starts.append(start)

    return _crop_and_pad(image, mask, starts, patch_size)


def random_patch(
    image: torch.Tensor,
    mask: torch.Tensor,
    patch_size: tuple[int, int, int] = PATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Crop a patch_size patch from a uniformly random location -- unlike
    foreground_patch, no guarantee it contains any of the mask's foreground (may
    land fully empty, or clip the foreground at a patch boundary). Counterpart to
    foreground_patch; see VoxTellFinetuneDataset's random_crop_fraction for how
    the two get mixed during training, and why: real sliding-window inference
    (evaluate_finetuned_voxtell_ed.py's sliding_window_predict) hands the model
    plenty of tiles like this, but foreground_patch alone never trains on one, so
    the model never learns to predict "no match" for its own text prompt in an
    unrelated patch, or to segment a mask that's cut off at the patch boundary.

    image: (C, D, H, W) float32.  mask: (D, H, W) float32 (0/1).
    """
    vol_shape = image.shape[1:]
    starts = [random.randint(0, size - p) if size > p else 0 for size, p in zip(vol_shape, patch_size)]
    return _crop_and_pad(image, mask, starts, patch_size)


class VoxTellFinetuneDataset(Dataset):
    """Same manifest schema as GrounderDataset (image/mask/sentence), but
    loading native-resolution volumes and cropping fixed 192^3 patches
    instead of resizing/padding to Grounder's own grid."""

    def __init__(
        self,
        manifest_path: str,
        image_dir: str,
        mask_dir: str,
        multi_window: bool = False,
        deterministic_crop: bool = False,
        max_samples: int | None = None,
        embedding_cache: str = "",
        random_crop_fraction: float = 0.0,
        patch_size: tuple[int, int, int] = PATCH_SIZE,
    ) -> None:
        with open(manifest_path) as f:
            data = json.load(f)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.samples = self._filter_missing(data)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.multi_window = multi_window
        self.deterministic_crop = deterministic_crop
        self.random_crop_fraction = random_crop_fraction
        self.patch_size = patch_size

        self._emb_index: dict[str, int] | None = None
        self._emb_feats = None
        if embedding_cache:
            cache_dir = Path(embedding_cache)
            with open(cache_dir / "index.json") as f:
                self._emb_index = json.load(f)
            self._emb_feats = np.load(str(cache_dir / "text_embeds.npy"), mmap_mode="r")
            missing = {s["mask"] for s in self.samples if s["mask"] not in self._emb_index}
            if missing:
                kept = [s for s in self.samples if s["mask"] not in missing]
                logger.warning(
                    f"Dropped {len(self.samples) - len(kept)}/{len(self.samples)} sample(s) in "
                    f"{manifest_path} missing from embedding cache {cache_dir} (e.g. {list(missing)[:5]}) "
                    f"-- rebuild the cache with precompute_voxtell_embeddings.py against the current "
                    f"manifest(s) to include them."
                )
                self.samples = kept

    def _filter_missing(self, samples: list[dict]) -> list[dict]:
        kept = [
            s for s in samples
            if (self.image_dir / s["image"]).exists() and (self.mask_dir / s["mask"]).exists()
        ]
        if len(kept) < len(samples):
            logger.warning(
                f"Dropped {len(samples) - len(kept)}/{len(samples)} sample(s) missing native "
                f"image/mask under {self.image_dir} / {self.mask_dir}"
            )
        return kept

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = self.image_dir / sample["image"]
        mask_path = self.mask_dir / sample["mask"]
        try:
            image_np = load_nifti_canonical(str(image_path))
            mask_np = load_nifti_canonical(str(mask_path))
        except Exception:
            logger.warning(f"Failed to load {sample['mask']}; skipping to next sample", exc_info=True)
            return self.__getitem__((idx + 1) % len(self.samples))

        image_np = apply_windows(image_np, self.multi_window)
        image = torch.from_numpy(image_np).float()
        mask = torch.from_numpy((mask_np > 0.5).astype(np.float32))

        if self.random_crop_fraction > 0 and random.random() < self.random_crop_fraction:
            image_patch, mask_patch = random_patch(image, mask, patch_size=self.patch_size)
        else:
            image_patch, mask_patch = foreground_patch(
                image, mask, patch_size=self.patch_size, deterministic=self.deterministic_crop
            )
        item = {"image": image_patch, "mask": mask_patch, "id": sample["mask"]}
        if self._emb_index is not None:
            item["text_embedding"] = torch.from_numpy(self._emb_feats[self._emb_index[sample["mask"]]].copy())
        else:
            item["sentence"] = sample["sentence"]
        return item


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "id": [b["id"] for b in batch],
    }
    if "text_embedding" in batch[0]:
        out["text_embedding"] = torch.stack([b["text_embedding"] for b in batch]).unsqueeze(1)  # (B, 1, D)
    else:
        out["sentence"] = [b["sentence"] for b in batch]
    return out


def build_voxtell_model(
    model_dir: Path,
    num_input_channels: int,
    from_scratch: bool = False,
    num_maskformer_stages: int = 5,
    decoder_layer: int = 4,
    text_embedding_dim: int = 2560,
) -> nn.Module:
    plans = json.loads((model_dir / "plans.json").read_text())
    arch = plans["configurations"]["3d_fullres"]["architecture"]
    arch_kwargs = dict(arch["arch_kwargs"])
    for key in arch["_kw_requires_import"]:
        if arch_kwargs[key] is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

    # decoder_configs.json (optional sibling of plans.json): VoxTellModel.DECODER_CONFIGS
    # is a hardcoded class-attribute table in the external voxtell package (channels/shape
    # per encoder stage index, sizing the frozen positional-encoding buffer and the
    # text->channel projection layers) with no constructor override. A config whose
    # arch_kwargs.features_per_stage doesn't match the package's own built-in table
    # (e.g. wider channels than the published checkpoint's) needs the table replaced to
    # match -- monkey-patched onto the class right before construction, below. Only
    # applied if this model_dir actually ships an override; every other config (including
    # voxtell_scratch_96, which deliberately keeps features_per_stage[0:5] identical to
    # the published checkpoint's so it can reuse DECODER_CONFIGS unmodified) is unaffected.
    decoder_configs_path = model_dir / "decoder_configs.json"
    if decoder_configs_path.exists():
        override = {int(k): v for k, v in json.loads(decoder_configs_path.read_text()).items()}
        for cfg in override.values():
            cfg["shape"] = tuple(cfg["shape"])
        logger.info(f"Overriding VoxTellModel.DECODER_CONFIGS from {decoder_configs_path}: {override}")
        VoxTellModel.DECODER_CONFIGS = override

    # decoder_layer/num_maskformer_stages select into VoxTellModel.DECODER_CONFIGS (see
    # above) -- when from_scratch=False, these two plus text_embedding_dim/num_heads/
    # query_dim/project_to_decoder_hidden_dim must match the published voxtell_v1.1
    # checkpoint exactly (see voxtell/inference/predictor.py's own VoxTellModel(...)
    # call), or the state dict below won't line up. With from_scratch=True, all of these
    # are free to vary together with model_dir's plans.json arch_kwargs (and
    # decoder_configs.json, if present), subject to DECODER_CONFIGS' per-index
    # (channels, shape) constraint -- see voxtell_scratch_96/plans.json's _description
    # for one worked derivation that reuses the table unmodified, and
    # voxtell_scratch_96_xl/decoder_configs.json for one that replaces it.
    model = VoxTellModel(
        input_channels=num_input_channels,
        **arch_kwargs,
        decoder_layer=decoder_layer,
        text_embedding_dim=text_embedding_dim,
        num_maskformer_stages=num_maskformer_stages,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
    )

    if from_scratch:
        logger.info("--from-scratch: keeping VoxTellModel's random init, not loading the pretrained checkpoint")
        return model

    checkpoint = torch.load(
        str(model_dir / "fold_0" / "checkpoint_final.pth"), map_location="cpu", weights_only=False
    )
    state_dict = dict(checkpoint["network_weights"])

    if num_input_channels != 1:
        for key in STEM_CONV_KEYS:
            if key in state_dict and state_dict[key].shape[1] == 1:
                w = state_dict[key]
                state_dict[key] = w.repeat(1, num_input_channels, 1, 1, 1) / num_input_channels
                logger.info(f"Adapted pretrained stem conv '{key}': 1 -> {num_input_channels} input channel(s)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint doesn't match VoxTellModel's architecture: "
            f"{len(missing)} missing key(s), {len(unexpected)} unexpected key(s)\n"
            f"missing={missing}\nunexpected={unexpected}"
        )
    return model


def load_text_backbone(text_encoder_path: str, device: torch.device) -> tuple[AutoTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, padding_side="left")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone = AutoModel.from_pretrained(text_encoder_path, dtype=dtype).eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    return tokenizer, backbone


@torch.no_grad()
def embed_sentences(
    tokenizer: AutoTokenizer, backbone: nn.Module, sentences: list[str], device: torch.device
) -> torch.Tensor:
    """(B,) sentences -> (B, 1, text_dim) embeddings, matching VoxTellPredictor's
    own instruction-wrapping/last-token-pooling exactly so the pretrained
    text-conditioning stays in-distribution.

    Deliberately @torch.no_grad() rather than @torch.inference_mode() (which
    predictor.py uses): the result feeds into project_text_embed's learnable
    Linear layers inside the training graph below, and inference-mode tensors
    can't be saved for that Linear's backward pass -- no_grad() still yields
    plain (non-inference) tensors that are safe to reuse this way."""
    prompts = [s.lower() for s in sentences]
    wrapped = wrap_with_instruction(prompts)
    tokens = tokenizer(wrapped, padding=True, truncation=True, max_length=8192, return_tensors="pt")
    tokens = {k: v.to(device) for k, v in tokens.items()}
    out = backbone(**tokens)
    emb = last_token_pool(out.last_hidden_state, tokens["attention_mask"]).float()
    return emb.unsqueeze(1)  # (B, 1, D)


def build_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Distributed training (same pattern as train.py/training/trainer.py)
# ---------------------------------------------------------------------------

def _is_ddp() -> bool:
    return "LOCAL_RANK" in os.environ


def _setup_ddp() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def _teardown_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _reduce_metrics(
    total_loss: float,
    total_dice: float,
    total_iou: float,
    total_hits: float,
    n: int,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    if world_size > 1:
        t = torch.tensor(
            [total_loss, total_dice, total_iou, total_hits, float(n)],
            dtype=torch.float64, device=device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss, total_dice, total_iou, total_hits, n = t.tolist()
    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
        "hit_rate": total_hits / n,
    }


def run_epoch(
    model: nn.Module,
    tokenizer: AutoTokenizer | None,
    text_backbone: nn.Module | None,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float,
    epoch: int,
    desc: str,
    is_main: bool = True,
) -> tuple[float, float, float, float, int]:
    """Returns raw (total_loss, total_dice, total_iou, total_hits, n) -- not
    yet averaged, since under DDP each rank only sees its own shard and these
    need to be all-reduced across ranks (see _reduce_metrics) before dividing."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_dice = total_iou = total_hits = 0.0
    n = 0

    for batch in tqdm(loader, desc=f"{desc} {epoch}", leave=False, disable=not is_main):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if "text_embedding" in batch:
            text_embedding = batch["text_embedding"].to(device, non_blocking=True)
        else:
            text_embedding = embed_sentences(tokenizer, text_backbone, batch["sentence"], device)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(image, text_embedding)
                loss, _ = loss_fn(logits, mask)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()

        bs = image.size(0)
        d = dice_score(logits.detach(), mask, from_logits=True)
        total_loss += loss.item() * bs
        total_dice += d.sum().item()
        total_iou += iou_score(logits.detach(), mask, from_logits=True).sum().item()
        total_hits += (d >= 0.1).sum().item()
        n += bs

    return total_loss, total_dice, total_iou, total_hits, n


@torch.no_grad()
def dice_per_sample(
    model: nn.Module,
    tokenizer: AutoTokenizer | None,
    text_backbone: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    is_main: bool = True,
) -> list[tuple[str, float]]:
    """(sample_id, dice) for every sample in `loader` that THIS rank sees --
    callers needing a global result must all_reduce, same as macro_hit_rate_epoch
    does below. Mirrors training/trainer.py's Trainer._dice_per_sample, adapted to
    VoxTellModel's (image, text_embedding) forward signature."""
    model.eval()
    results: list[tuple[str, float]] = []
    for batch in tqdm(loader, desc=desc, leave=False, disable=not is_main):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if "text_embedding" in batch:
            text_embedding = batch["text_embedding"].to(device, non_blocking=True)
        else:
            text_embedding = embed_sentences(tokenizer, text_backbone, batch["sentence"], device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(image, text_embedding)
        d = dice_score(logits, mask, from_logits=True)
        results.extend(zip(batch["id"], d.tolist()))
    return results


@torch.no_grad()
def macro_hit_rate_epoch(
    model: nn.Module,
    tokenizer: AutoTokenizer | None,
    text_backbone: nn.Module | None,
    ed_loader: DataLoader | None,
    onc_loader: DataLoader | None,
    ed_category_by_mask: dict[str, str],
    device: torch.device,
    world_size: int,
    epoch: int,
    is_main: bool = True,
) -> dict[str, float]:
    """Same definition as training/trainer.py's Trainer.macro_hit_rate_epoch: ONC's
    is the plain hit rate (dice >= 0.1) across all ONC val samples; ED's is the MEAN
    of the per-category hit rate across ED_CATEGORIES (13 categories weighted
    equally, not sample-weighted) -- see FINDING_TO_ED_CATEGORY. The reported
    macro_hit_rate is the average of those two. No-op ({}) if ed_loader/onc_loader
    weren't given (--ed-val-manifest/--onc-val-manifest "")."""
    if ed_loader is None or onc_loader is None:
        return {}

    onc_results = dice_per_sample(model, tokenizer, text_backbone, onc_loader, device,
                                   desc=f"ONC hit {epoch}", is_main=is_main)
    onc_hits = float(sum(1 for _, d in onc_results if d >= 0.1))
    onc_n = float(len(onc_results))

    ed_results = dice_per_sample(model, tokenizer, text_backbone, ed_loader, device,
                                  desc=f"ED hit {epoch}", is_main=is_main)
    cat_hits = torch.zeros(len(ED_CATEGORIES), dtype=torch.float64, device=device)
    cat_counts = torch.zeros(len(ED_CATEGORIES), dtype=torch.float64, device=device)
    for sample_id, d in ed_results:
        idx = ED_CATEGORIES.index(ed_category_by_mask[sample_id])
        cat_counts[idx] += 1
        if d >= 0.1:
            cat_hits[idx] += 1

    if world_size > 1:
        onc_t = torch.tensor([onc_hits, onc_n], dtype=torch.float64, device=device)
        dist.all_reduce(onc_t, op=dist.ReduceOp.SUM)
        onc_hits, onc_n = onc_t.tolist()
        dist.all_reduce(cat_hits, op=dist.ReduceOp.SUM)
        dist.all_reduce(cat_counts, op=dist.ReduceOp.SUM)

    if (cat_counts == 0).any():
        missing = [c for c, n in zip(ED_CATEGORIES, cat_counts.tolist()) if n == 0]
        raise RuntimeError(
            f"macro_hit_rate_epoch: no ED validation samples for categor{'y' if len(missing) == 1 else 'ies'} "
            f"{missing} -- check --ed-val-manifest"
        )

    onc_macro_hit_rate = onc_hits / onc_n
    ed_macro_hit_rate = (cat_hits / cat_counts).mean().item()
    return {
        "onc_macro_hit_rate": onc_macro_hit_rate,
        "ed_macro_hit_rate": ed_macro_hit_rate,
        "macro_hit_rate": (onc_macro_hit_rate + ed_macro_hit_rate) / 2.0,
    }


def log_metrics(metrics_log_path: Path, record: dict) -> None:
    """Append one JSON line per epoch, same metrics.jsonl convention as
    training/trainer.py's Trainer._log_json."""
    record = {"time": datetime.now(timezone.utc).isoformat(), **record}
    with open(metrics_log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(
    output_dir: Path, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
    scheduler, best_dice: float, is_best: bool, keep_last_n: int,
    best_macro_hit_rate: float, epochs_without_macro_improvement: int,
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_dice": best_dice,
        "best_macro_hit_rate": best_macro_hit_rate,
        "epochs_without_macro_improvement": epochs_without_macro_improvement,
    }, path)
    if is_best:
        best_path = ckpt_dir / "best.pt"
        shutil.copy2(path, best_path)
        logger.info(f"New best checkpoint (dice={best_dice:.4f}) -> {best_path}")

    checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in checkpoints[:-keep_last_n]:
        old.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if _is_ddp():
        rank, local_rank, world_size = _setup_ddp()
        device = torch.device("cuda", local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device(args.device)

    is_main = (rank == 0)
    if not is_main:
        logging.disable(logging.CRITICAL)

    num_channels = 3 if args.multi_window else 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = output_dir / "metrics.jsonl"

    if args.from_scratch:
        logger.info(f"Building VoxTell model (input_channels={num_channels}) with random init (--from-scratch)")
    else:
        logger.info(f"Building VoxTell model (input_channels={num_channels}) from {args.model_dir}")
    raw_model = build_voxtell_model(
        Path(args.model_dir), num_channels, from_scratch=args.from_scratch,
        num_maskformer_stages=args.num_maskformer_stages, decoder_layer=args.decoder_layer,
        text_embedding_dim=args.text_embedding_dim,
    ).to(device)
    if world_size > 1:
        raw_model = SyncBatchNorm.convert_sync_batchnorm(raw_model)
        # find_unused_parameters=True: VoxTellModel is built with deep_supervision=False
        # (see build_voxtell_model), which suggests some decoder-stage parameters may not
        # route to the single output we use every forward pass -- the model's own source
        # isn't vendored into this repo to verify, so this errs safe against a DDP crash
        # rather than assuming every parameter always gets a gradient.
        model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        model = raw_model

    if args.embedding_cache:
        logger.info(f"Using precomputed text embedding cache: {args.embedding_cache} (text backbone not loaded)")
        tokenizer, text_backbone = None, None
    else:
        logger.info(f"Loading frozen text backbone {args.text_encoder}")
        tokenizer, text_backbone = load_text_backbone(args.text_encoder, device)

    train_ds = VoxTellFinetuneDataset(
        args.train_manifest, args.image_dir, args.mask_dir,
        multi_window=args.multi_window,
        deterministic_crop=False, max_samples=args.max_samples,
        embedding_cache=args.embedding_cache,
        random_crop_fraction=args.random_crop_fraction,
        patch_size=args.patch_size,
    )
    # random_crop_fraction is deliberately NOT passed below -- val/held-out splits
    # stay 100% foreground-guaranteed/deterministic (see --random-crop-fraction's
    # help), so Macro Hit Rate/early stopping keep a low-variance signal epoch to
    # epoch instead of fluctuating with which random crops happened to be drawn.
    val_ds = VoxTellFinetuneDataset(
        args.val_manifest, args.image_dir, args.mask_dir,
        multi_window=args.multi_window,
        deterministic_crop=True, max_samples=args.max_samples,
        embedding_cache=args.embedding_cache,
        patch_size=args.patch_size,
    )

    # Held-out ED/ONC splits for Macro Hit Rate / early stopping (see module
    # docstring) -- both must be set or neither is used, same convention as
    # training/trainer.py's data.ed_val_manifest/data.onc_val_manifest.
    if args.ed_val_manifest and args.onc_val_manifest:
        ed_val_ds = VoxTellFinetuneDataset(
            args.ed_val_manifest, args.image_dir, args.mask_dir,
            multi_window=args.multi_window,
            deterministic_crop=True, max_samples=args.max_samples,
            embedding_cache=args.embedding_cache,
            patch_size=args.patch_size,
        )
        onc_val_ds = VoxTellFinetuneDataset(
            args.onc_val_manifest, args.image_dir, args.mask_dir,
            multi_window=args.multi_window,
            deterministic_crop=True, max_samples=args.max_samples,
            embedding_cache=args.embedding_cache,
            patch_size=args.patch_size,
        )
        ed_category_by_mask = {s["mask"]: FINDING_TO_ED_CATEGORY[s["finding"]] for s in ed_val_ds.samples}
    else:
        ed_val_ds = onc_val_ds = None
        ed_category_by_mask = {}

    if world_size > 1:
        # drop_last=True on both the sampler and the DataLoader: every rank must see the
        # same number of batches per epoch, or DDP's gradient sync hangs waiting on a
        # rank that ran out of batches first.
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                            shuffle=True, drop_last=True, seed=args.seed)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank,
                                          shuffle=False, drop_last=False)
        ed_val_sampler = (DistributedSampler(ed_val_ds, num_replicas=world_size, rank=rank,
                                              shuffle=False, drop_last=False) if ed_val_ds else None)
        onc_val_sampler = (DistributedSampler(onc_val_ds, num_replicas=world_size, rank=rank,
                                               shuffle=False, drop_last=False) if onc_val_ds else None)
    else:
        train_sampler = None
        val_sampler = None
        ed_val_sampler = None
        onc_val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        drop_last=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        drop_last=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    ed_val_loader = DataLoader(
        ed_val_ds, batch_size=args.batch_size, shuffle=False, sampler=ed_val_sampler,
        drop_last=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    ) if ed_val_ds else None
    onc_val_loader = DataLoader(
        onc_val_ds, batch_size=args.batch_size, shuffle=False, sampler=onc_val_sampler,
        drop_last=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    ) if onc_val_ds else None
    logger.info(
        f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}  "
        f"ED samples: {len(ed_val_ds) if ed_val_ds else 0}  ONC samples: {len(onc_val_ds) if onc_val_ds else 0}  "
        f"world_size={world_size}"
    )

    loss_fn = CombinedLoss(dice_weight=0.5, bce_weight=0.5, bce_pos_weight=10.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.constant_lr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        scheduler = build_scheduler(optimizer, args.num_epochs, args.warmup_epochs)

    start_epoch = 0
    best_dice = 0.0
    best_macro_hit_rate = 0.0
    epochs_without_macro_improvement = 0
    if args.resume:
        # Always load into raw_model (the unwrapped module) -- save_checkpoint() below
        # saves raw_model.state_dict() (no "module." prefix), which would fail to load
        # into a DDP-wrapped model directly.
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"], strict=not args.resume_partial)
        if args.resume_partial:
            logger.info(f"Warm-started weights from {args.resume} (strict=False); optimizer/scheduler/epoch not restored")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            best_dice = ckpt.get("best_dice", 0.0)
            best_macro_hit_rate = ckpt.get("best_macro_hit_rate", 0.0)
            epochs_without_macro_improvement = ckpt.get("epochs_without_macro_improvement", 0)
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")
    if world_size > 1:
        dist.barrier()

    try:
        for epoch in range(start_epoch, args.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            lr = optimizer.param_groups[0]["lr"]
            train_totals = run_epoch(
                model, tokenizer, text_backbone, train_loader, loss_fn, device,
                optimizer, args.grad_clip_norm, epoch, desc="Train", is_main=is_main,
            )
            train_metrics = _reduce_metrics(*train_totals, device, world_size)

            val_totals = run_epoch(
                model, tokenizer, text_backbone, val_loader, loss_fn, device,
                None, args.grad_clip_norm, epoch, desc="Val", is_main=is_main,
            )
            val_metrics = _reduce_metrics(*val_totals, device, world_size)

            macro_metrics = macro_hit_rate_epoch(
                model, tokenizer, text_backbone, ed_val_loader, onc_val_loader,
                ed_category_by_mask, device, world_size, epoch, is_main=is_main,
            )
            scheduler.step()

            # should_stop/epochs_without_macro_improvement computed on every rank
            # (macro_metrics is already identical on every rank, all-reduced inside
            # macro_hit_rate_epoch), so every rank breaks the loop together below --
            # no extra broadcast needed. Unlike this, best_dice/checkpoint writing
            # stays rank-0-only, since it doesn't affect loop control flow.
            should_stop = False
            if macro_metrics:
                if macro_metrics["macro_hit_rate"] > best_macro_hit_rate:
                    best_macro_hit_rate = macro_metrics["macro_hit_rate"]
                    epochs_without_macro_improvement = 0
                else:
                    epochs_without_macro_improvement += 1
                if args.early_stop_patience and epochs_without_macro_improvement >= args.early_stop_patience:
                    should_stop = True

            if is_main:
                log_line = (
                    f"Epoch {epoch} | lr={lr:.2e} | "
                    f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} hit={train_metrics['hit_rate']:.3f} | "
                    f"val   loss={val_metrics['loss']:.4f}  dice={val_metrics['dice']:.4f}  hit={val_metrics['hit_rate']:.3f}"
                )
                if macro_metrics:
                    log_line += (
                        f" | macro_hit={macro_metrics['macro_hit_rate']:.3f} "
                        f"(ed={macro_metrics['ed_macro_hit_rate']:.3f} onc={macro_metrics['onc_macro_hit_rate']:.3f})"
                    )
                logger.info(log_line)
                if should_stop:
                    logger.info(
                        f"Early stopping: macro_hit_rate hasn't improved for "
                        f"{epochs_without_macro_improvement} epoch(s) "
                        f"(patience={args.early_stop_patience}, best={best_macro_hit_rate:.3f})"
                    )

                is_best = val_metrics["dice"] > best_dice
                if is_best:
                    best_dice = val_metrics["dice"]
                log_metrics(metrics_log_path, {
                    "epoch": epoch,
                    "lr": lr,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"val_{k}": v for k, v in macro_metrics.items()},
                })
                save_checkpoint(output_dir, epoch, raw_model, optimizer, scheduler, best_dice, is_best,
                                 args.keep_last_n, best_macro_hit_rate, epochs_without_macro_improvement)
            if world_size > 1:
                dist.barrier()
            if should_stop:
                break
    finally:
        _teardown_ddp()


if __name__ == "__main__":
    main()
