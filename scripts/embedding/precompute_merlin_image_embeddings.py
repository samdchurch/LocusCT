"""
Precompute and cache Merlin (StanfordMIMI/Merlin CT foundation model)
image-only embeddings for every unique CT volume referenced by the
configured manifests.

Not to be confused with precompute_merlin_embeddings.py, which caches text
embeddings for the (unrelated) Merlin Abdominal CT Dataset's report
sentences -- this script instead runs our own CT volumes through the
pretrained Merlin *model* (github.com/StanfordMIMI/Merlin) to get one
2048-dim image embedding per volume.

Run once:
    python precompute_merlin_image_embeddings.py --config configs/default.yaml \
        --merlin-model-dir /path/to/models/Merlin --output /path/to/merlin_image_embeddings_mmap

The output directory contains:
    image_feats.npy           (N, 2048) float32 -- pooled embedding, memory-mappable
    image_feats_prepool.npy   (N, 2048, D', H', W') float32 -- layer4 feature map
                              before Merlin's own avgpool, memory-mappable (D'/H'/W'
                              depend on Merlin's fixed input size; captured via a
                              forward hook since I3ResNet.forward has no flag to skip
                              pooling)
    index.json                {sample["image"] value: row_index} -- shared by both arrays

Merlin.__init__ always tries to download its checkpoint from Hugging Face
Hub to a path relative to the installed `merlin` package (not overridable
via argument or env var). The container filesystem is read-only, so this
script can't stage the checkpoint there itself -- the submission script
bind-mounts --merlin-model-dir directly onto that expected in-package path
instead; this script only verifies the checkpoint is visible there before
instantiating Merlin() (see ensure_checkpoint_staged and the project plan
for the full rationale).

Merlin expects its own preprocessing (RAS orientation, 1.5x1.5x3mm spacing,
HU clip [-1000, 1000] rescaled to [0, 1], center-cropped/padded to
224x224x160) via merlin.data.monai_transforms.ImageTransforms -- this is
unrelated to GrounderDataset's own HU-window/spatial-mode handling, so raw
NIfTI paths are fed through Merlin's transforms directly rather than going
through GrounderDataset.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import monai
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from models.merlin_utils import (
    disable_resnet152_imagenet_download,
    ensure_checkpoint_staged,
    redirect_clinical_longformer,
)
from train import apply_overrides

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", required=True, help="Output directory for memmap cache")
    parser.add_argument(
        "--merlin-model-dir", required=True,
        help="Directory containing Merlin's pretrained checkpoint file (staged from Hugging Face Hub ahead of time)",
    )
    parser.add_argument(
        "--clinical-longformer-dir", required=True,
        help="Local directory holding a save_pretrained() copy of yikuan8/Clinical-Longformer "
             "(staged ahead of time -- this cluster has no internet access, and merlin's "
             "TextEncoder hardcodes that Hugging Face Hub repo id with no override)",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dot-notation config overrides, e.g. data.train_manifest=[...]",
    )
    return parser.parse_args()


def register_prepool_hook(model) -> dict:
    """Capture I3ResNet's layer4 feature map before Merlin's own avgpool.

    I3ResNet.forward (merlin/models/i3res.py) always applies self.avgpool before
    returning in ImageEmbedding mode, with no flag to skip it -- a forward
    pre-hook on avgpool is the only way to get the pre-pool feature map without
    reimplementing the forward pass. Returns a dict that's overwritten with the
    latest batch's captured tensor (dict["feat"]) on every forward call; read it
    immediately after each model(...) call, before the next one overwrites it.
    """
    captured: dict = {}

    def _capture(module, inputs):
        captured["feat"] = inputs[0]

    model.model.encode_image.i3_resnet.avgpool.register_forward_pre_hook(_capture)
    return captured


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Using device: {device}")

    ensure_checkpoint_staged(args.merlin_model_dir)
    disable_resnet152_imagenet_download()
    redirect_clinical_longformer(args.clinical_longformer_dir)

    from merlin import Merlin
    from merlin.data.monai_transforms import ImageTransforms

    model = Merlin(ImageEmbedding=True).to(device)
    model.eval()
    prepool_capture = register_prepool_hook(model)

    # Collect unique image paths from all configured manifests -- Merlin embeds
    # volumes, not (image, mask, expression) triples, so dedupe by image path.
    manifest_paths: list[str] = []
    for key in ("train_manifest", "val_manifest", "test_manifest"):
        v = cfg["data"].get(key)
        if v is None:
            continue
        manifest_paths.extend([v] if isinstance(v, str) else v)

    image_dir = Path(cfg["data"]["image_dir"]) if cfg["data"].get("image_dir") else None
    images: dict[str, str] = {}  # manifest image value (id) -> resolved full path
    for path in manifest_paths:
        with open(path) as f:
            data = json.load(f)
        for s in data:
            image_val = s["image"]
            if image_val in images:
                continue
            full_path = str(image_dir / image_val) if image_dir else image_val
            images[image_val] = full_path

    N = len(images)
    logger.info(f"Computing Merlin embeddings for {N} unique CT volume(s) across {len(manifest_paths)} manifests")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats_mm = np.lib.format.open_memmap(
        str(out_dir / "image_feats.npy"), mode="w+", dtype=np.float32, shape=(N, 2048)
    )

    image_ids = list(images.keys())
    datalist = [{"image": images[image_id]} for image_id in image_ids]
    dataset = monai.data.Dataset(data=datalist, transform=ImageTransforms)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=monai.data.list_data_collate,
    )

    # prepool_mm's per-volume shape isn't known until the first forward pass reveals
    # it (via prepool_capture) -- Merlin's input size is fixed, so it's the same for
    # every volume, and it's preallocated once that first shape is seen.
    prepool_mm = None

    index: dict[str, int] = {}
    row = 0
    with torch.no_grad(), tqdm(total=N, desc="Encoding", unit="volume") as pbar:
        for batch in loader:
            outputs = model(batch["image"].to(device))
            batch_feats = outputs[0].float().cpu().numpy()
            batch_prepool = prepool_capture["feat"].float().cpu().numpy()
            n = batch_feats.shape[0]

            if prepool_mm is None:
                prepool_shape = (N,) + batch_prepool.shape[1:]
                prepool_mm = np.lib.format.open_memmap(
                    str(out_dir / "image_feats_prepool.npy"), mode="w+",
                    dtype=np.float32, shape=prepool_shape,
                )
                logger.info(f"Pre-pool feature map shape per volume: {batch_prepool.shape[1:]}")

            feats_mm[row : row + n] = batch_feats
            prepool_mm[row : row + n] = batch_prepool
            for i in range(n):
                index[image_ids[row + i]] = row + i
            row += n
            pbar.update(n)

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f)

    feats_mm.flush()
    prepool_mm.flush()
    size_mb = feats_mm.nbytes / (1024 ** 2)
    prepool_size_mb = prepool_mm.nbytes / (1024 ** 2)
    logger.info(
        f"Saved {N} embeddings, shape {feats_mm.shape} ({feats_mm.dtype}), {size_mb:.1f} MB, "
        f"and prepool features, shape {prepool_mm.shape} ({prepool_mm.dtype}), "
        f"{prepool_size_mb:.1f} MB, to {out_dir}"
    )


if __name__ == "__main__":
    main()
