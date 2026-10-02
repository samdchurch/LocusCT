#!/usr/bin/env python3
"""
Run a trained Grounder checkpoint over the Merlin abdominal CT dataset's
per-study atomic findings (merlin_sentences.json, one JSON object per line:
{"study_id": ..., "atomic_findings": [{"finding", "organ", "laterality",
"status"}, ...]}), producing a predicted segmentation mask for each
(study, finding) pair whose status is "present" (default -- see --status),
excluding findings that just say normal/unremarkable/patent (see
is_normal_finding, --include-normal to keep them) since there's nothing to
localize for those. There's no ground truth here, just raw inference over
Merlin's own report-derived text; see visualize_merlin_predictions.py to
look at the results.

study_id maps to its resampled volume as "<image-dir>/<study_id>.nii.gz"
(flat, no subfolders). Studies with no such file are skipped with a warning.

Only runs inference on a random --n sample of (study, finding) pairs (default
20, --n 0 for everything), seeded by --seed -- this is meant to feed
visualize_merlin_predictions.py's QC dump, so there's no point spending GPU
time (and the 8B-parameter text encoder, if not using --embedding-cache) on
predictions nobody's going to look at.

Text encoding has two modes, selected by --embedding-cache:

  Not given (default): text is live-tokenized and run through the real text
  encoder (load_text_backbone=True) -- pass
  --override model.text_encoder_name=/path/to/local/Qwen3-Embedding-8B if the
  config's default (a HF hub id) isn't reachable from the compute node.
  Loading the checkpoint therefore uses strict=False: a checkpoint trained
  with data.embedding_cache set never saves text_encoder.* weights at all
  (train.py never loads that backbone in that mode), so those keys are
  expected to be missing here. That's only safe because the text encoder is
  frozen (model.finetune_last_n_layers=0) -- the freshly-loaded pretrained
  weights are then identical to whatever would've been saved anyway. This
  script refuses to run (see main()) if that's not the case, since silently
  proceeding would use un-fine-tuned weights for supposedly fine-tuned layers.

  Given: text_feats/text_padding_mask are looked up from a cache directory
  built by precompute_merlin_embeddings.py (same format as
  precompute_embeddings.py's training-time cache: text_feats.npy,
  text_padding_mask.npy, index.json, keyed by sample_key(study_id,
  finding_idx)). The text encoder is never loaded (load_text_backbone=False)
  and the checkpoint loads with strict=True -- this is the mode to use for
  any checkpoint actually trained with data.embedding_cache set (i.e. most
  of them, since that's this repo's default), and is much faster since the
  8B-parameter frozen backbone never runs. The cache must have been built
  with matching --sentences/--image-dir/--status, or some requested samples
  won't be in its index (raises a clear error listing how many, rather than
  a raw KeyError per missing sample).

Predicted masks are saved as .nii.gz with a placeholder identity affine (not
the source scan's real affine) in the same canonical (D, H, W) array
orientation load_nifti_canonical produces for the source image -- so they
line up index-for-index with a fresh load_nifti_canonical(image_path) call,
but do NOT re-run load_nifti_canonical (i.e. nib.as_closest_canonical) on
the saved mask file itself, which would incorrectly re-transpose it (same
gotcha noted in evaluate.py's _save_masks).

Usage
-----
    python predict_merlin.py --config configs/default.yaml --checkpoint runs/default/checkpoints/best.pt
    python predict_merlin.py --config configs/default.yaml --checkpoint best.pt --n 50 --output-dir /tmp/merlin_test
    python predict_merlin.py --config configs/default.yaml --checkpoint best.pt --n 0  # run on everything
    python predict_merlin.py --config configs/default.yaml --checkpoint best.pt \
        --category-findings-dir merlin_ed_category_findings --output-dir outputs/eval/merlin_ed_categories

--category-findings-dir runs inference directly on a directory of per-ED-category
*.json files (see merlin_ed_category_findings/), each a list of {"exam": study_id,
"phrase": finding text} objects -- each phrase IS the referring expression, used as-is
(--sentences/--status/--include-normal are ignored; --embedding-cache can't be
combined with this mode, since these samples were never part of merlin_sentences.json,
which precompute_merlin_embeddings.py's cache is keyed from -- live text encoding is
used instead). There is deliberately NO cross-reference against merlin_sentences.json:
its "finding" field is a separately LLM-normalized paraphrase of the same report
sentence, and only ~2% of the exams these category files reference even exist as a
study_id there -- an earlier version of this script tried matching phrase text against
it and found the overlap essentially unusable, not a text-normalization bug to patch.
Overrides --n/--seed sampling entirely; --max-per-category (default 50, 0 = no cap)
still applies, capping each category independently to a random subset of that many
entries (seeded by --seed) -- an entry dropped by one category's cap can still be kept
via another category referencing the identical (exam, phrase) pair. Results are saved
under <output-dir>/<category>/predicted_masks and <output-dir>/<category>/
predictions.json -- an (exam, phrase) pair referenced by more than one category's file
is saved (inference run once, output copied) into each of those category folders.
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from data.dataset import MULTI_WINDOWS, load_nifti_canonical
from models.grounder import Grounder
from train import apply_overrides

DEFAULT_SENTENCES = Path(__file__).resolve().parents[2] / "reference_data" / "merlin_sentences.json"
DEFAULT_IMAGE_DIR = Path("/path/to/data/inhouse_abdominal_ct/public_datasets/merlinabdominalctdataset/merlin_data_resampled")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval/merlin_predictions"))
    parser.add_argument("--status", nargs="+", default=["present"],
                         help="Only run inference on atomic_findings with one of these status values (default: present)")
    parser.add_argument("--include-normal", action="store_true",
                         help="Include findings that just say normal/unremarkable/patent (see is_normal_finding) -- "
                              "skipped by default since there's nothing to localize for those")
    parser.add_argument("--embedding-cache", type=Path, default=None,
                         help="Directory built by precompute_merlin_embeddings.py -- if given, text_feats/"
                              "text_padding_mask are looked up from it instead of live-tokenizing (see module docstring)")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to the config's training.batch_size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n", type=int, default=20,
                         help="Randomly sample this many (study, finding) pairs to run inference on -- "
                              "matches visualize_merlin_predictions.py's own --n (default: 20; 0 = everything). "
                              "Ignored entirely when --category-findings-dir is given.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--category-findings-dir", type=Path, default=None,
        help="Directory of per-ED-category *.json files (each a list of {'exam','phrase'} "
             "objects, e.g. merlin_ed_category_findings/) -- runs directly on each phrase as "
             "the referring expression, with NO merlin_sentences.json cross-reference (see "
             "module docstring for why). Overrides --n/--seed/--sentences/--status/"
             "--include-normal; incompatible with --embedding-cache.",
    )
    parser.add_argument(
        "--max-per-category", type=int, default=50,
        help="Cap each ed-category to at most this many findings (random subset, seeded by "
             "--seed) -- only applies with --category-findings-dir. 0 = no cap.",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Dot-notation config overrides, e.g. model.text_encoder_name=/path/to/model",
    )
    return parser.parse_args()


def sample_key(study_id: str, finding_idx: int) -> str:
    """Unique id per (study, finding) sample -- shared between predicted-mask filenames
    here and precompute_merlin_embeddings.py's embedding-cache index keys, so the two
    scripts always agree without needing to duplicate the format string."""
    return f"{study_id}_finding{finding_idx}"


# Findings whose text amounts to "this looked normal" -- nothing to localize/ground.
# Merlin's reports also template these per-organ as "<Organ>: Normal." (e.g. "Pancreas:
# Normal.", "Vasculature: Patent."), hence stripping an optional leading "<word(s)>: ".
_NORMAL_FINDING_PHRASES = {
    "normal", "unremarkable", "wnl", "within normal limits", "grossly normal",
    "patent", "no abnormality", "no acute abnormality", "unremarkable study",
    "negative", "no significant abnormality",
}
_ORGAN_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z /]{0,30}:\s*")


def is_normal_finding(text: str) -> bool:
    """Whether `text` is just "normal"/"unremarkable"/"patent"/etc. (optionally
    prefixed with Merlin's "<Organ>: " template) rather than an actual finding."""
    stripped = text.strip().rstrip(".").strip()
    without_prefix = _ORGAN_PREFIX_RE.sub("", stripped)
    return stripped.lower() in _NORMAL_FINDING_PHRASES or without_prefix.lower() in _NORMAL_FINDING_PHRASES


def load_merlin_samples(
    sentences_path: Path, image_dir: Path, statuses: list[str], skip_normal: bool = True
) -> list[dict]:
    """One entry per (study, finding) whose status is in `statuses` and whose resampled
    volume exists on disk. Studies missing that volume are skipped (warned once, not
    per-finding). Findings that are just "normal"/"unremarkable"/"patent" (see
    is_normal_finding) are skipped too unless skip_normal=False."""
    samples = []
    n_studies = 0
    n_missing_image = 0
    n_normal_skipped = 0
    with open(sentences_path) as f:
        for line in f:
            rec = json.loads(line)
            n_studies += 1
            study_id = rec["study_id"]
            image_path = image_dir / f"{study_id}.nii.gz"
            if not image_path.exists():
                n_missing_image += 1
                continue
            for i, finding in enumerate(rec.get("atomic_findings", [])):
                if finding.get("status") not in statuses:
                    continue
                sentence = (finding.get("finding") or "").strip()
                if not sentence:
                    continue
                if skip_normal and is_normal_finding(sentence):
                    n_normal_skipped += 1
                    continue
                samples.append({
                    "study_id": study_id,
                    "finding_idx": i,
                    "sentence": sentence,
                    "organ": finding.get("organ", ""),
                    "laterality": finding.get("laterality", ""),
                })
    if n_missing_image:
        logger.warning(f"Skipped {n_missing_image}/{n_studies} study/studies with no volume under {image_dir}")
    if n_normal_skipped:
        logger.info(f"Skipped {n_normal_skipped} normal/unremarkable/patent finding(s) (see is_normal_finding)")
    logger.info(f"Loaded {len(samples)} (study, finding) sample(s) from {n_studies} studies, status in {statuses}")
    return samples


def load_ed_category_findings(category_dir: Path) -> dict[str, list[dict]]:
    """{category_name: [{"exam": study_id, "phrase": finding text}, ...]} from every
    *.json file under category_dir (see merlin_ed_category_findings/) -- category name
    is each file's stem."""
    categories: dict[str, list[dict]] = {}
    for path in sorted(category_dir.glob("*.json")):
        with open(path) as f:
            categories[path.stem] = json.load(f)
    return categories


def load_ed_category_samples(
    category_dir: Path,
    image_dir: Path,
    max_per_category: int | None = None,
    seed: int = 0,
) -> tuple[list[dict], dict[str, set[str]]]:
    """Build samples directly from merlin_ed_category_findings/*.json's own (exam,
    phrase) entries -- each phrase IS the referring expression run through the model,
    with NO cross-reference against merlin_sentences.json. (Matching against it was
    tried and found fundamentally broken: only ~2% of the exams these category files
    reference even exist as a study_id there, and even for those that do, its
    "finding" field is a separately LLM-normalized paraphrase of the same report
    sentence, not the same text -- fewer than 10% of even that overlapping subset
    matched exactly. Not a text-normalization bug to patch; a different, much
    smaller/differently-derived corpus.)

    If max_per_category is set, each category independently keeps at most that many
    entries (a random subset, seeded by `seed`) -- capped per category, so an entry
    dropped by one category's cap can still be kept via another category referencing
    the identical (exam, phrase) pair.

    Returns (samples, {sample_key: {category names}}). Each sample dict has the same
    shape load_merlin_samples produces (study_id, finding_idx, sentence, organ,
    laterality) so MerlinInferenceDataset needs no changes -- finding_idx is a
    synthetic per-exam counter (stable, sorted-order assignment for reproducibility),
    not one recovered from merlin_sentences.json. organ/laterality are always ""
    (not present in these category files).
    """
    categories = load_ed_category_findings(category_dir)
    rng = random.Random(seed)

    # Per-category: de-dup identical (exam, phrase) repeats within one file, then cap.
    pair_categories: dict[tuple[str, str], set[str]] = {}
    for category, entries in categories.items():
        seen: set[tuple[str, str]] = set()
        keys: list[tuple[str, str]] = []
        for entry in entries:
            key = (entry["exam"], entry["phrase"].strip())
            if key not in seen:
                seen.add(key)
                keys.append(key)
        if max_per_category is not None and len(keys) > max_per_category:
            keys = rng.sample(keys, max_per_category)
        logger.info(f"  [{category}] {len(keys)} finding(s) selected")
        for key in keys:
            pair_categories.setdefault(key, set()).add(category)

    # Merge across categories (the same (exam, phrase) pair referenced by two category
    # files becomes ONE sample tagged with both, not two separate ones), assign a
    # stable synthetic finding_idx per unique pair, and drop entries with no volume.
    finding_idx_by_exam: dict[str, int] = {}
    samples: list[dict] = []
    sample_categories: dict[str, set[str]] = {}
    n_missing_image = 0
    for exam, phrase in sorted(pair_categories):
        image_path = image_dir / f"{exam}.nii.gz"
        if not image_path.exists():
            n_missing_image += 1
            continue
        idx = finding_idx_by_exam.get(exam, 0)
        finding_idx_by_exam[exam] = idx + 1
        samples.append({"study_id": exam, "finding_idx": idx, "sentence": phrase, "organ": "", "laterality": ""})
        sample_categories[sample_key(exam, idx)] = pair_categories[(exam, phrase)]

    if n_missing_image:
        logger.warning(f"Skipped {n_missing_image} ed-category finding(s) with no volume under {image_dir}")
    logger.info(
        f"Built {len(samples)} sample(s) directly from {len(categories)} ed-category file(s)' own phrase text"
        + (f", capped at {max_per_category} per category" if max_per_category is not None else "")
    )
    return samples, sample_categories


def apply_windows(vol: np.ndarray, hu_min: float, hu_max: float, multi_window: bool) -> np.ndarray:
    """Mirrors GrounderDataset._apply_windows -- normalize HU values, returns (C, D, H, W)."""
    if multi_window:
        channels = []
        for lo, hi in MULTI_WINDOWS:
            c = np.clip(vol, lo, hi)
            c = (c - lo) / (hi - lo) * 2.0 - 1.0
            channels.append(c)
        return np.stack(channels, axis=0)
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min) * 2.0 - 1.0
    return vol[np.newaxis]


def pad_to_divisible(image_np: np.ndarray, divisor: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirrors GrounderDataset._pad_to_divisible, image only (no mask to pad here)."""
    D, H, W = image_np.shape[-3:]
    pad_D = (divisor - D % divisor) % divisor
    pad_H = (divisor - H % divisor) % divisor
    pad_W = (divisor - W % divisor) % divisor
    image = torch.from_numpy(image_np)
    padding = (0, pad_W, 0, pad_H, 0, pad_D)
    image = F.pad(image.float(), padding, mode="constant", value=-1.0)
    pad_amounts = torch.tensor([pad_D, pad_H, pad_W], dtype=torch.long)
    return image, pad_amounts


def resize_volume(vol: np.ndarray, target: tuple[int, int, int]) -> torch.Tensor:
    """Mirrors GrounderDataset._resize_volume, image only."""
    t = torch.from_numpy(vol).float().unsqueeze(0)  # (1, C, D, H, W)
    t = F.interpolate(t, size=target, mode="trilinear", align_corners=False)
    return t.squeeze(0)


def trim_padding(logits: torch.Tensor, pad_amounts: torch.Tensor) -> torch.Tensor:
    """Single-tensor variant of training.trainer._trim_padding (no GT mask to trim here)."""
    pa = pad_amounts[0] if pad_amounts.dim() == 2 else pad_amounts
    pad_D, pad_H, pad_W = pa[0].item(), pa[1].item(), pa[2].item()
    if pad_D == 0 and pad_H == 0 and pad_W == 0:
        return logits
    D = logits.shape[2] - pad_D if pad_D > 0 else logits.shape[2]
    H = logits.shape[3] - pad_H if pad_H > 0 else logits.shape[3]
    W = logits.shape[4] - pad_W if pad_W > 0 else logits.shape[4]
    return logits[:, :, :D, :H, :W]


class MerlinInferenceDataset(Dataset):
    """Image + text (live-tokenized, or looked up from a precomputed embedding cache --
    see embedding_cache below), no mask -- mirrors GrounderDataset's preprocessing for
    the parts that apply here (see apply_windows/pad_to_divisible/resize_volume)."""

    def __init__(
        self,
        samples: list[dict],
        image_dir: Path,
        tokenizer_name: str,
        spatial_size: tuple[int, int, int],
        max_text_len: int,
        hu_min: float,
        hu_max: float,
        multi_window: bool,
        spatial_mode: str,
        embedding_cache: Path | None = None,
    ) -> None:
        self.samples = samples
        self.image_dir = image_dir
        self.spatial_size = spatial_size
        self.max_text_len = max_text_len
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.multi_window = multi_window
        self.spatial_mode = spatial_mode

        self._emb_index: dict[str, int] | None = None
        if embedding_cache:
            with open(embedding_cache / "index.json") as f:
                self._emb_index = json.load(f)
            self._emb_feats = np.load(str(embedding_cache / "text_feats.npy"), mmap_mode="r")
            self._emb_masks = np.load(str(embedding_cache / "text_padding_mask.npy"), mmap_mode="r")
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = self.image_dir / f"{sample['study_id']}.nii.gz"
        image_np = load_nifti_canonical(str(image_path))
        image_np = apply_windows(image_np, self.hu_min, self.hu_max, self.multi_window)

        if self.spatial_mode == "fixed":
            image, pad_amounts = pad_to_divisible(image_np)
        else:
            image = resize_volume(image_np, self.spatial_size)
            pad_amounts = torch.zeros(3, dtype=torch.long)

        item = {
            "image": image,
            "pad_amounts": pad_amounts,
            "study_id": sample["study_id"],
            "finding_idx": sample["finding_idx"],
            "sentence": sample["sentence"],
            "organ": sample["organ"],
            "laterality": sample["laterality"],
        }

        if self._emb_index is not None:
            row = self._emb_index[sample_key(sample["study_id"], sample["finding_idx"])]
            item["text_feats"] = torch.from_numpy(self._emb_feats[row].copy())
            item["text_padding_mask"] = torch.from_numpy(self._emb_masks[row].copy())
        else:
            encoding = self.tokenizer(
                sample["sentence"], max_length=self.max_text_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            item["input_ids"] = encoding["input_ids"].squeeze(0)
            item["attention_mask"] = encoding["attention_mask"].squeeze(0)

        return item


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    use_cached_text = bool(args.embedding_cache)
    if args.category_findings_dir and use_cached_text:
        raise ValueError(
            "--embedding-cache can't be used with --category-findings-dir -- those samples "
            "are built directly from the category files' own phrase text, not from "
            "merlin_sentences.json, so they can never be in a merlin_sentences.json-keyed "
            "embedding cache. Omit --embedding-cache (live text encoding will be used)."
        )
    if not use_cached_text and cfg["model"].get("finetune_last_n_layers", 0) > 0:
        raise ValueError(
            "This config has model.finetune_last_n_layers > 0, meaning the text encoder's "
            "top layers were actually fine-tuned. Loading the checkpoint with strict=False "
            "(needed for live text encoding since Merlin sentences have no precomputed "
            "embedding_cache entry by default) would silently keep pretrained, not "
            "fine-tuned, weights for those layers -- not supported. Use --embedding-cache "
            "instead (see precompute_merlin_embeddings.py), which loads with strict=True."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = cfg["inference"].get("threshold", 0.5)

    model = Grounder(
        text_encoder_name=cfg["model"]["text_encoder_name"],
        text_proj_dim=cfg["model"]["text_proj_dim"],
        freeze_text_encoder=True,
        finetune_last_n_layers=0,
        unet_base_channels=cfg["model"]["unet_base_channels"],
        unet_channel_mult=cfg["model"]["unet_channel_mult"],
        num_heads=cfg["model"]["num_heads"],
        target_q_tokens=cfg["model"]["target_q_tokens"],
        dropout=0.0,
        in_channels=3 if cfg["data"].get("multi_window") else 1,
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        fusion_type=cfg["model"].get("fusion_type", "cross_attention"),
        voxtell_guidance_dim=cfg["model"].get("voxtell", {}).get("guidance_dim", 32),
        voxtell_prompt_decoder_dim=cfg["model"].get("voxtell", {}).get("prompt_decoder_dim", 256),
        voxtell_prompt_decoder_layers=cfg["model"].get("voxtell", {}).get("prompt_decoder_layers", 6),
        voxtell_prompt_decoder_heads=cfg["model"].get("voxtell", {}).get("prompt_decoder_heads", 8),
        load_text_backbone=not use_cached_text,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if use_cached_text:
        # No text_encoder submodule exists at all in this mode, so the checkpoint (saved
        # the same way if it was trained with data.embedding_cache set) should match exactly.
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        missing_non_text = [k for k in missing if not k.startswith("text_encoder.")]
        if missing_non_text:
            raise RuntimeError(f"Checkpoint is missing non-text-encoder weight(s), can't proceed: {missing_non_text}")
        if unexpected:
            raise RuntimeError(f"Checkpoint has unexpected key(s): {unexpected}")
        logger.info(f"{len(missing)} text_encoder.* key(s) left at pretrained init, as expected")
    model.eval()
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    sample_categories: dict[str, set[str]] | None = None  # sample_key -> {category names}
    if args.category_findings_dir:
        samples, sample_categories = load_ed_category_samples(
            args.category_findings_dir, args.image_dir,
            max_per_category=args.max_per_category or None, seed=args.seed,
        )
    else:
        samples = load_merlin_samples(args.sentences, args.image_dir, args.status, skip_normal=not args.include_normal)
        if args.n:
            rng = random.Random(args.seed)
            samples = rng.sample(samples, min(args.n, len(samples)))
            logger.info(f"Randomly sampled {len(samples)} sample(s) for inference (seed={args.seed}) "
                        f"-- matches what visualize_merlin_predictions.py will show")
    if not samples:
        logger.error("No samples to run -- check --sentences/--image-dir/--status/--category-findings-dir")
        sys.exit(1)

    if use_cached_text:
        with open(args.embedding_cache / "index.json") as f:
            cache_index = json.load(f)
        missing_keys = [s for s in samples if sample_key(s["study_id"], s["finding_idx"]) not in cache_index]
        if missing_keys:
            raise RuntimeError(
                f"{len(missing_keys)}/{len(samples)} requested sample(s) are missing from "
                f"{args.embedding_cache} -- rebuild it with matching --sentences/--image-dir/--status "
                f"(e.g. first missing: {missing_keys[0]['study_id']} finding {missing_keys[0]['finding_idx']})"
            )

    dataset = MerlinInferenceDataset(
        samples=samples,
        image_dir=args.image_dir,
        tokenizer_name=cfg["model"]["text_encoder_name"],
        spatial_size=tuple(cfg["data"]["spatial_size"]),
        max_text_len=cfg["data"]["max_text_len"],
        hu_min=cfg["data"]["hu_min"],
        hu_max=cfg["data"]["hu_max"],
        multi_window=cfg["data"].get("multi_window", False),
        spatial_mode=cfg["data"]["spatial_mode"],
        embedding_cache=args.embedding_cache,
    )
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    if sample_categories is not None:
        category_names = sorted({c for cats in sample_categories.values() for c in cats})
        masks_dirs = {c: args.output_dir / c / "predicted_masks" for c in category_names}
        for d in masks_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        results_by_category: dict[str, list[dict]] = {c: [] for c in category_names}
    else:
        masks_dir = args.output_dir / "predicted_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            image = batch["image"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if use_cached_text:
                    text_feats = batch["text_feats"].to(device)
                    text_padding_mask = batch["text_padding_mask"].to(device)
                    logits = model(image, text_feats=text_feats, text_padding_mask=text_padding_mask)
                else:
                    input_ids = batch["input_ids"].to(device)
                    attn_mask = batch["attention_mask"].to(device)
                    logits = model(image, input_ids, attn_mask)
            logits = trim_padding(logits, batch["pad_amounts"])
            probs = torch.sigmoid(logits).cpu().float().numpy()

            for i, study_id in enumerate(batch["study_id"]):
                finding_idx = int(batch["finding_idx"][i])
                pred_np = (probs[i, 0] > threshold).astype(np.uint8)
                mask_filename = f"{sample_key(study_id, finding_idx)}.nii.gz"
                record_base = {
                    "study_id": study_id,
                    "finding_idx": finding_idx,
                    "sentence": batch["sentence"][i],
                    "organ": batch["organ"][i],
                    "laterality": batch["laterality"][i],
                    "image_path": str(args.image_dir / f"{study_id}.nii.gz"),
                    "voxel_count": int(pred_np.sum()),
                }

                if sample_categories is not None:
                    # Inference ran once for this sample even if it matches multiple
                    # categories -- save the same prediction into each matching folder.
                    sk = sample_key(study_id, finding_idx)
                    for category in sorted(sample_categories[sk]):
                        mask_path = masks_dirs[category] / mask_filename
                        nib.save(nib.Nifti1Image(pred_np, affine=np.eye(4)), str(mask_path))
                        results_by_category[category].append({**record_base, "mask_path": str(mask_path)})
                else:
                    mask_path = masks_dir / mask_filename
                    nib.save(nib.Nifti1Image(pred_np, affine=np.eye(4)), str(mask_path))
                    results.append({**record_base, "mask_path": str(mask_path)})

    if sample_categories is not None:
        total = 0
        for category, cat_results in results_by_category.items():
            predictions_path = args.output_dir / category / "predictions.json"
            with open(predictions_path, "w") as f:
                json.dump(cat_results, f, indent=2)
            n_empty = sum(1 for r in cat_results if r["voxel_count"] == 0)
            logger.info(f"[{category}] Wrote {len(cat_results)} prediction(s) -> {predictions_path} "
                        f"({n_empty} empty)")
            total += len(cat_results)
        logger.info(f"Wrote {total} total prediction(s) across {len(results_by_category)} "
                    f"category folder(s) under {args.output_dir}")
    else:
        predictions_path = args.output_dir / "predictions.json"
        with open(predictions_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Wrote {len(results)} prediction(s) -> {predictions_path}")
        logger.info(f"  {sum(1 for r in results if r['voxel_count'] == 0)}/{len(results)} predicted an empty mask")


if __name__ == "__main__":
    main()
