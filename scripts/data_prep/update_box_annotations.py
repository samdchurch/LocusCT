"""Replace line-derived mask pairs in official_splits with their merged box-annotation mask.

For each "Converting to box: accession X, series S, slice_idx I, anno_idx A and B" line in
conversions.log, this deletes the two split entries whose masks are
{accession}/mask_{S}_{I}_{A}.nii.gz and {accession}/mask_{S}_{I}_{B}.nii.gz, and inserts a single
entry pointing at {accession}/mask_box_{S}_{I}_0.nii.gz in ED_LABELS_BOX. Skips conversions whose
box mask file doesn't exist yet, or whose two source entries aren't both present in a given split.
"""

import argparse
import json
import re
from pathlib import Path

CONVERSION_RE = re.compile(
    r"Converting to box: accession ([^,]+), series (\d+), slice_idx (\d+), "
    r"anno_idx (\d+) and (\d+)"
)


def parse_conversions(log_path: Path):
    conversions = []
    for line in log_path.read_text().splitlines():
        m = CONVERSION_RE.match(line.strip())
        if not m:
            continue
        accession, series, slice_idx, anno_a, anno_b = m.groups()
        conversions.append((accession, series, slice_idx, anno_a, anno_b))
    return conversions


def mask_rel_path(accession: str, series: str, slice_idx: str, anno: str) -> str:
    return f"{accession}/mask_{series}_{slice_idx}_{anno}.nii.gz"


def box_mask_rel_path(accession: str, series: str, slice_idx: str) -> str:
    return f"{accession}/mask_box_{series}_{slice_idx}_0.nii.gz"


def apply_conversions_to_split(entries, conversions, box_masks_dir: Path, log_entries: list):
    mask_to_index = {e["mask"]: i for i, e in enumerate(entries)}

    for accession, series, slice_idx, anno_a, anno_b in conversions:
        box_mask_rel = box_mask_rel_path(accession, series, slice_idx)
        if not (box_masks_dir / box_mask_rel).exists():
            continue

        mask_a = mask_rel_path(accession, series, slice_idx, anno_a)
        mask_b = mask_rel_path(accession, series, slice_idx, anno_b)
        idx_a = mask_to_index.get(mask_a)
        idx_b = mask_to_index.get(mask_b)
        if idx_a is None or idx_b is None:
            continue

        entry_a = entries[idx_a]
        entry_b = entries[idx_b]

        sentence_a = entry_a["sentence"]
        sentence_b = entry_b["sentence"]
        if sentence_a == sentence_b:
            sentence = sentence_a
        else:
            sentence = f"{sentence_a} {sentence_b}"
            log_entries.append(
                {
                    "accession": accession,
                    "series": series,
                    "slice_idx": slice_idx,
                    "anno_idx": [anno_a, anno_b],
                    "sentence_a": sentence_a,
                    "sentence_b": sentence_b,
                    "joined_sentence": sentence,
                }
            )

        new_entry = {
            "image": entry_a["image"],
            "mask": box_mask_rel,
            "sentence": sentence,
            "region": entry_a["region"],
            "finding": entry_a["finding"],
        }

        insert_at = min(idx_a, idx_b)
        for i in sorted([idx_a, idx_b], reverse=True):
            del entries[i]
        entries.insert(insert_at, new_entry)

        mask_to_index = {e["mask"]: i for i, e in enumerate(entries)}

    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversions_log",
        type=Path,
        default=Path("/path/to/data/inhouse_abdominal_ct/deid_gsps_box/conversions.log"),
    )
    parser.add_argument(
        "--box_masks_dir",
        type=Path,
        default=Path("/path/to/data/inhouse_abdominal_ct/ED_LABELS_BOX"),
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "official_splits",
    )
    parser.add_argument(
        "--mismatch_log",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reference_data" / "box_conversion_sentence_mismatches.json",
    )
    args = parser.parse_args()

    conversions = parse_conversions(args.conversions_log)
    print(f"Parsed {len(conversions)} conversions from {args.conversions_log}")

    log_entries: list = []
    for split_path in sorted(args.splits_dir.glob("*.json")):
        entries = json.loads(split_path.read_text())
        before = len(entries)
        entries = apply_conversions_to_split(entries, conversions, args.box_masks_dir, log_entries)
        after = len(entries)
        split_path.write_text(json.dumps(entries, indent=4))
        print(f"{split_path.name}: {before} -> {after} entries")

    if log_entries:
        args.mismatch_log.write_text(json.dumps(log_entries, indent=4))
        print(f"Logged {len(log_entries)} sentence mismatches to {args.mismatch_log}")


if __name__ == "__main__":
    main()
