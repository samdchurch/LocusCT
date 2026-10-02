"""Apply corrected joined_sentence values from box_conversion_sentence_mismatches.json
to the matching mask_box_* records in official_splits/*.json.

box_conversion_sentence_mismatches.json is produced by update_box_annotations.py and
identifies each merged box annotation by (accession, series, slice_idx), whose mask is
{accession}/mask_box_{series}_{slice_idx}_0.nii.gz in the split files. After manually
editing the "joined_sentence" fields in that log, run this script to push the corrected
sentences back into the split files.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MISMATCHES_PATH = REPO_ROOT / "box_conversion_sentence_mismatches.json"
SPLITS_DIR = REPO_ROOT / "official_splits"


def box_mask_rel_path(accession: str, series: str, slice_idx: str) -> str:
    return f"{accession}/mask_box_{series}_{slice_idx}_0.nii.gz"


def build_update_map(mismatches: list[dict]) -> dict[str, str]:
    sentences_by_mask: dict[str, set] = {}
    for entry in mismatches:
        mask = box_mask_rel_path(entry["accession"], entry["series"], entry["slice_idx"])
        sentences_by_mask.setdefault(mask, set()).add(entry["joined_sentence"])

    update_map = {}
    for mask, sentences in sentences_by_mask.items():
        if len(sentences) > 1:
            print(f"WARNING: conflicting joined_sentence values for {mask}, skipping:")
            for s in sentences:
                print(f"    - {s!r}")
            continue
        update_map[mask] = next(iter(sentences))
    return update_map


def update_split_file(path: Path, update_map: dict[str, str]) -> int:
    records = json.loads(path.read_text(encoding="utf-8"))

    updated = 0
    for record in records:
        joined_sentence = update_map.get(record["mask"])
        if joined_sentence is None or record["sentence"] == joined_sentence:
            continue
        record["sentence"] = joined_sentence
        updated += 1

    if updated:
        path.write_text(json.dumps(records, indent=4), encoding="utf-8")
    return updated


def main() -> None:
    mismatches = json.loads(MISMATCHES_PATH.read_text(encoding="utf-8"))
    update_map = build_update_map(mismatches)

    total = 0
    for split_path in sorted(SPLITS_DIR.glob("*.json")):
        updated = update_split_file(split_path, update_map)
        total += updated
        print(f"{split_path.name}: updated {updated} record(s)")

    print(f"Total updated: {total}")


if __name__ == "__main__":
    main()
