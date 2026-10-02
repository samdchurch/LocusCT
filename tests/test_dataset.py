import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from data.dataset import GrounderImageGroupedDataset, build_dataloader

VOL_SHAPE = (8, 8, 8)  # padded up to (16,16,16) by _pad_group_to_divisible (divisor=16)
TEXT_LEN = 4
TEXT_DIM = 6


def _save_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4)), str(path))


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "images").mkdir()
    (root / "masks").mkdir()

    # img_a: valid, 4 findings -- a3's mask is corrupted, a4's mask shape
    # doesn't match the image's (a1/a2 are the only ones that survive)
    _save_nifti(root / "images" / "img_a.nii.gz", np.zeros(VOL_SHAPE))
    _save_nifti(root / "masks" / "mask_a1.nii.gz", np.ones(VOL_SHAPE))
    _save_nifti(root / "masks" / "mask_a2.nii.gz", np.ones(VOL_SHAPE))
    (root / "masks" / "mask_a3.nii.gz").write_bytes(b"not a real nifti file")
    _save_nifti(root / "masks" / "mask_a4.nii.gz", np.ones((4, 4, 4)))  # mismatched shape

    # img_b: corrupted image -- whole group skipped, mask_b1 never surfaces
    (root / "images" / "img_b.nii.gz").write_bytes(b"not a real nifti file")
    _save_nifti(root / "masks" / "mask_b1.nii.gz", np.ones(VOL_SHAPE))

    # img_c: valid, single finding
    _save_nifti(root / "images" / "img_c.nii.gz", np.zeros(VOL_SHAPE))
    _save_nifti(root / "masks" / "mask_c1.nii.gz", np.ones(VOL_SHAPE))

    # img_d: valid, 5 findings, all valid -- for max_findings_per_image capping
    _save_nifti(root / "images" / "img_d.nii.gz", np.zeros(VOL_SHAPE))
    for i in range(1, 6):
        _save_nifti(root / "masks" / f"mask_d{i}.nii.gz", np.ones(VOL_SHAPE))

    # img_missing/mask_missing: neither file created -- dropped by the
    # inherited GrounderDataset._filter_missing before grouping ever runs.

    manifest = [
        {"image": "images/img_a.nii.gz", "mask": "masks/mask_a1.nii.gz", "sentence": "finding a1"},
        {"image": "images/img_a.nii.gz", "mask": "masks/mask_a2.nii.gz", "sentence": "finding a2"},
        {"image": "images/img_a.nii.gz", "mask": "masks/mask_a3.nii.gz", "sentence": "finding a3"},
        {"image": "images/img_a.nii.gz", "mask": "masks/mask_a4.nii.gz", "sentence": "finding a4"},
        {"image": "images/img_b.nii.gz", "mask": "masks/mask_b1.nii.gz", "sentence": "finding b1"},
        {"image": "images/img_c.nii.gz", "mask": "masks/mask_c1.nii.gz", "sentence": "finding c1"},
        {"image": "images/img_missing.nii.gz", "mask": "masks/mask_missing.nii.gz", "sentence": "finding missing"},
    ] + [
        {"image": "images/img_d.nii.gz", "mask": f"masks/mask_d{i}.nii.gz", "sentence": f"finding d{i}"}
        for i in range(1, 6)
    ]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # Embedding cache -- skips loading a real tokenizer entirely, keyed by
    # each entry's "mask" (relative) path, matching GrounderDataset's convention.
    emb_dir = root / "emb_cache"
    emb_dir.mkdir()
    mask_keys = [m["mask"] for m in manifest if "missing" not in m["mask"]]
    index = {k: i for i, k in enumerate(mask_keys)}
    (emb_dir / "index.json").write_text(json.dumps(index))
    np.save(emb_dir / "text_feats.npy", np.random.randn(len(mask_keys), TEXT_LEN, TEXT_DIM).astype(np.float32))
    np.save(emb_dir / "text_padding_mask.npy", np.zeros((len(mask_keys), TEXT_LEN), dtype=bool))

    return root


def _build_dataset(root: Path, **kwargs) -> GrounderImageGroupedDataset:
    return GrounderImageGroupedDataset(
        manifest_path=str(root / "manifest.json"),
        tokenizer_name="unused",  # embedding_cache set below, so no real tokenizer is loaded
        spatial_size=(16, 16, 16),
        image_dir=str(root),
        mask_dir=str(root),
        embedding_cache=str(root / "emb_cache"),
        **kwargs,
    )


def test_groups_by_unique_image_and_filters_missing(fixture_dir: Path) -> None:
    dataset = _build_dataset(fixture_dir)
    # img_missing's sample is dropped by the inherited _filter_missing before
    # grouping -- only img_a, img_b, img_c, img_d remain as groups.
    assert len(dataset) == 4


def test_drops_unreadable_mask_keeps_rest_of_group(fixture_dir: Path) -> None:
    dataset = _build_dataset(fixture_dir)
    items = [dataset[i] for i in range(len(dataset))]
    all_ids = [mid for item in items for mid in item["id"]]

    assert "masks/mask_a1.nii.gz" in all_ids
    assert "masks/mask_a2.nii.gz" in all_ids
    assert "masks/mask_a3.nii.gz" not in all_ids  # corrupted mask, dropped
    assert "masks/mask_a4.nii.gz" not in all_ids  # shape mismatch vs image, dropped

    a_item = next(item for item in items if "masks/mask_a1.nii.gz" in item["id"])
    assert sorted(a_item["id"]) == ["masks/mask_a1.nii.gz", "masks/mask_a2.nii.gz"]
    assert a_item["mask"].shape[0] == 2
    assert a_item["image"].shape[0] == 1


def test_skips_group_with_unreadable_image(fixture_dir: Path) -> None:
    dataset = _build_dataset(fixture_dir)
    items = [dataset[i] for i in range(len(dataset))]
    all_ids = [mid for item in items for mid in item["id"]]

    # img_b's own load fails -> the whole group is skipped (recurses to the
    # next index) -- mask_b1 never surfaces in any returned item.
    assert "masks/mask_b1.nii.gz" not in all_ids
    assert "masks/mask_c1.nii.gz" in all_ids


def test_item_shapes(fixture_dir: Path) -> None:
    dataset = _build_dataset(fixture_dir)
    items = [dataset[i] for i in range(len(dataset))]

    for item in items:
        n = item["mask"].shape[0]
        assert item["image"].shape == (1, 1, 16, 16, 16)  # padded 8 -> 16 (divisor=16)
        assert item["mask"].shape == (n, 1, 16, 16, 16)
        assert len(item["id"]) == n
        assert item["text_feats"].shape == (n, TEXT_LEN, TEXT_DIM)
        assert item["text_padding_mask"].shape == (n, TEXT_LEN)
        assert item["pad_amounts"].shape == (3,)


def test_build_dataloader_group_by_image(fixture_dir: Path) -> None:
    """Regression test: DataLoader(batch_size=None, drop_last=True) raises
    ValueError ("batch_size=None option ... mutually exclusive with
    drop_last") -- single-process train build_dataloader sets drop_last=True
    by default, so this must be forced off when group_by_image is on."""
    cfg = {
        "data": {
            "image_dir": str(fixture_dir),
            "mask_dir": str(fixture_dir),
            "embedding_cache": str(fixture_dir / "emb_cache"),
            "spatial_size": [16, 16, 16],
            "max_text_len": TEXT_LEN,
            "hu_min": -1000.0,
            "hu_max": 1000.0,
            "multi_window": False,
            "spatial_mode": "fixed",
            "num_workers": 0,
            "pin_memory": False,
            "group_by_image": True,
            "augment_train": False,
            "max_samples": None,
        },
        "training": {"batch_size": 1},
        "model": {"text_encoder_name": "unused"},
    }
    loader = build_dataloader(str(fixture_dir / "manifest.json"), cfg, split="train", num_workers=0)
    batches = list(loader)
    assert len(batches) == 4  # img_a, img_b (skipped -> falls through to img_c), img_c, img_d
    for batch in batches:
        assert batch["image"].shape[0] == 1
        assert batch["mask"].shape[0] == len(batch["id"])


def test_max_findings_per_image_caps_group_size(fixture_dir: Path) -> None:
    dataset = _build_dataset(fixture_dir, max_findings_per_image=2)
    d_index = next(i for i in range(len(dataset)) if dataset[i]["id"][0].startswith("masks/mask_d"))

    d_item = dataset[d_index]
    assert len(d_item["id"]) == 2
    assert d_item["mask"].shape[0] == 2

    # Re-sampled per call (not cached), so across enough draws we see more
    # than one distinct subset of img_d's 5 findings, not the same 2 every time.
    seen = {tuple(sorted(dataset[d_index]["id"])) for _ in range(30)}
    assert len(seen) > 1


def test_max_findings_per_image_none_uses_all(fixture_dir: Path) -> None:
    dataset = _build_dataset(fixture_dir, max_findings_per_image=None)
    d_item = next(
        item for item in (dataset[i] for i in range(len(dataset)))
        if item["id"][0].startswith("masks/mask_d")
    )
    assert len(d_item["id"]) == 5


@pytest.fixture
def merlin_fixture_dir(tmp_path: Path) -> Path:
    """Small enough that Merlin's own fixed-grid resampling/pad is cheap --
    the point of this fixture is just to confirm build_merlin_grid_transform
    always lands on the exact fixed (1,224,224,160) output shape, not to
    exercise any particular input size."""
    root = tmp_path
    (root / "images").mkdir()
    (root / "masks").mkdir()
    _save_nifti(root / "images" / "img_a.nii.gz", np.zeros((8, 8, 8)))
    _save_nifti(root / "masks" / "mask_a1.nii.gz", np.ones((8, 8, 8)))

    manifest = [
        {"image": "images/img_a.nii.gz", "mask": "masks/mask_a1.nii.gz", "sentence": "finding a1"},
    ]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    emb_dir = root / "emb_cache"
    emb_dir.mkdir()
    index = {"masks/mask_a1.nii.gz": 0}
    (emb_dir / "index.json").write_text(json.dumps(index))
    np.save(emb_dir / "text_feats.npy", np.random.randn(1, TEXT_LEN, TEXT_DIM).astype(np.float32))
    np.save(emb_dir / "text_padding_mask.npy", np.zeros((1, TEXT_LEN), dtype=bool))

    return root


def test_merlin_grid_dataset_fixed_output_shape(merlin_fixture_dir: Path) -> None:
    pytest.importorskip("monai")
    from data.merlin_dataset import MerlinGridDataset

    dataset = MerlinGridDataset(
        manifest_path=str(merlin_fixture_dir / "manifest.json"),
        tokenizer_name="unused",  # embedding_cache set below, so no real tokenizer is loaded
        image_dir=str(merlin_fixture_dir),
        mask_dir=str(merlin_fixture_dir),
        embedding_cache=str(merlin_fixture_dir / "emb_cache"),
    )
    assert len(dataset) == 1
    item = dataset[0]

    assert item["image"].shape == (1, 224, 224, 160)
    assert item["mask"].shape == (1, 224, 224, 160)
    assert torch.equal(item["pad_amounts"], torch.zeros(3, dtype=torch.long))
    assert item["id"] == "masks/mask_a1.nii.gz"
    assert item["text_feats"].shape == (TEXT_LEN, TEXT_DIM)
    # mask values stay binary after nearest-neighbor resample + re-binarize
    assert set(item["mask"].unique().tolist()) <= {0.0, 1.0}
