#!/usr/bin/env python3
"""
Checks, for every (image, mask) pair referenced by one or more manifests,
whether the two files' on-disk shapes actually match each other -- and
reports the overall shape distribution seen across all images and all masks.

Motivation: GrounderDataset's "fixed" spatial mode (data/dataset.py,
_pad_to_divisible) computes its padding amount from the IMAGE's shape only,
then applies that same padding to the mask. If an image and its paired mask
don't already share the same input shape (e.g. because a resample-grid
migration regenerated nifti_resampled/ without regenerating labels_resampled/
to match), the mask silently ends up at the wrong final depth -- which
surfaces downstream as a `torch.stack` collate RuntimeError, not here. This
script finds those mismatches directly, before training ever sees them.

Only reads NIfTI headers (img.shape), not voxel data, so it's fast enough to
run over the full dataset.

Usage
-----
    python check_data_shapes.py --manifest official_splits/all_data_train.json \\
        official_splits/curated_ed_onc_val_data.json official_splits/all_test_data.json \\
        --image-dir /path/to/data/inhouse_abdominal_ct/nifti_resampled \\
        --mask-dir /path/to/data/inhouse_abdominal_ct/labels_resampled
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-mismatches-shown", type=int, default=30)
    parser.add_argument("--out", default=None, help="Optional path to write full per-sample JSON results")
    return parser.parse_args()


def file_shape(path: Path) -> tuple[int, ...] | None:
    if not path.exists():
        return None
    return tuple(nib.load(str(path)).shape)


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)

    pairs = []  # (image_rel, mask_rel)
    seen_pairs = set()
    for manifest_path in args.manifest:
        with open(manifest_path) as f:
            samples = json.load(f)
        for s in samples:
            key = (s["image"], s["mask"])
            if key not in seen_pairs:
                seen_pairs.add(key)
                pairs.append(key)

    unique_images = sorted({image_rel for image_rel, _ in pairs})
    unique_masks = sorted({mask_rel for _, mask_rel in pairs})
    print(f"{len(pairs)} unique (image, mask) pairs across {len(args.manifest)} manifest(s)  "
          f"({len(unique_images)} unique images, {len(unique_masks)} unique masks)")

    image_shapes: dict[str, tuple[int, ...] | None] = {}
    mask_shapes: dict[str, tuple[int, ...] | None] = {}

    def load_all(rels: list[str], base: Path, out: dict) -> None:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(file_shape, base / rel): rel for rel in rels}
            done = 0
            for fut in as_completed(futures):
                rel = futures[fut]
                out[rel] = fut.result()
                done += 1
                if done % 2000 == 0:
                    print(f"  ...{done}/{len(rels)}")

    print("Reading image headers...")
    load_all(unique_images, image_dir, image_shapes)
    print("Reading mask headers...")
    load_all(unique_masks, mask_dir, mask_shapes)

    image_shape_counts = Counter(s for s in image_shapes.values() if s is not None)
    mask_shape_counts = Counter(s for s in mask_shapes.values() if s is not None)
    missing_images = [rel for rel, s in image_shapes.items() if s is None]
    missing_masks = [rel for rel, s in mask_shapes.items() if s is None]

    mismatches = []
    for image_rel, mask_rel in pairs:
        img_shape = image_shapes.get(image_rel)
        mask_shape = mask_shapes.get(mask_rel)
        if img_shape is None or mask_shape is None:
            continue
        if img_shape != mask_shape:
            mismatches.append((image_rel, mask_rel, img_shape, mask_shape))

    print()
    print("=" * 70)
    print("IMAGE shape distribution")
    print("=" * 70)
    for shape, count in image_shape_counts.most_common():
        print(f"  {count:6d}  {shape}")
    if missing_images:
        print(f"  {len(missing_images):6d}  MISSING")

    print()
    print("=" * 70)
    print("MASK shape distribution")
    print("=" * 70)
    for shape, count in mask_shape_counts.most_common():
        print(f"  {count:6d}  {shape}")
    if missing_masks:
        print(f"  {len(missing_masks):6d}  MISSING")

    print()
    print("=" * 70)
    print(f"IMAGE/MASK SHAPE MISMATCHES: {len(mismatches)} / {len(pairs)} pairs")
    print("=" * 70)
    for image_rel, mask_rel, img_shape, mask_shape in mismatches[: args.max_mismatches_shown]:
        print(f"  image={img_shape}  mask={mask_shape}   {image_rel}  <->  {mask_rel}")
    if len(mismatches) > args.max_mismatches_shown:
        print(f"  ... and {len(mismatches) - args.max_mismatches_shown} more")

    if missing_images:
        print()
        print(f"MISSING IMAGES: {len(missing_images)}")
        for rel in missing_images[: args.max_mismatches_shown]:
            print(f"  {rel}")

    if missing_masks:
        print()
        print(f"MISSING MASKS: {len(missing_masks)}")
        for rel in missing_masks[: args.max_mismatches_shown]:
            print(f"  {rel}")

    if args.out:
        result = {
            "n_pairs": len(pairs),
            "image_shape_counts": {str(k): v for k, v in image_shape_counts.items()},
            "mask_shape_counts": {str(k): v for k, v in mask_shape_counts.items()},
            "n_mismatches": len(mismatches),
            "mismatches": [
                {"image": i, "mask": m, "image_shape": ish, "mask_shape": msh}
                for i, m, ish, msh in mismatches
            ],
            "missing_images": missing_images,
            "missing_masks": missing_masks,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
