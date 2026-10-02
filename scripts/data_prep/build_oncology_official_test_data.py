#!/usr/bin/env python3
"""
Build official_splits/onc_official_test_data.json -- the oncology
counterpart to official_splits/ed_official_test_data.json -- from
OncologyTestSetWorksheet.xlsx (Sheet1), keeping only rows marked
"good" in the reviewer's disposition column.

That column has no text header in the worksheet (row 1 is merged group
headers, row 2 is the real column headers, but this particular column's
row-2 cell is blank in both) -- it's an informal status flag with values
drawn from {"good", "fixed", "remove"} (plus blank for not-yet-reviewed
rows). Since it can't be found by header name like the other columns, it's
auto-detected as the column whose non-null values are all in that small
set and include at least one "good" -- see _find_disposition_column().
Rows where it's anything other than exactly "good" (fixed, remove, blank,
or not present) are dropped.

Each kept row's mask path is reconstructed as
"{Accession}/mask_{Series}_{Slice}_{AnnoIdx}.nii.gz" (same convention as
make_results_figure.py / update_box_annotations.py's mask_rel_path). Image
path, region, and finding are looked up from official_splits/all_test_data.json
by that mask path, falling back to a plain series-file search under
--image-dir/{accession}/ if the mask isn't in that manifest. Rows missing
Accession/Series/Slice/AnnoIdx/Sentence, or whose image can't be resolved,
are skipped with a warning.

Requires pandas + openpyxl (pip install pandas openpyxl if missing).

Usage
-----
    python build_oncology_official_test_data.py \\
        --image-dir /path/to/data/inhouse_abdominal_ct/nifti_resampled
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

DEFAULT_WORKSHEET = Path(__file__).resolve().parents[2] / "reference_data" / "OncologyTestSetWorksheet.xlsx"
DEFAULT_ALL_TEST_MANIFEST = Path(__file__).resolve().parents[2] / "official_splits" / "all_test_data.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "official_splits" / "onc_official_test_data.json"
DISPOSITION_VALUES = {"good", "fixed", "remove"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def _normalize_col(s: str) -> str:
    return " ".join(str(s).split()).strip().lower()


def _find_disposition_column(df: pd.DataFrame) -> str:
    """The reviewer's good/fixed/remove status column has no header text, so it's
    identified by its values instead: the column whose non-null values are all in
    DISPOSITION_VALUES and include at least one "good"."""
    for col in df.columns:
        values = {str(v).strip().lower() for v in df[col].dropna()}
        if values and values.issubset(DISPOSITION_VALUES) and "good" in values:
            return col
    raise KeyError(
        "Couldn't find the good/fixed/remove disposition column -- no column's values "
        f"are a subset of {DISPOSITION_VALUES} including 'good'. Columns: {list(df.columns)}"
    )


def _find_series_file(accession_dir: Path, series: int) -> str | None:
    if not accession_dir.is_dir():
        return None
    for f in sorted(accession_dir.glob("*.nii.gz")):
        prefix = f.name.split("_")[0]
        if prefix.isdigit() and int(prefix) == series:
            return f.name
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    parser.add_argument("--all-test-manifest", type=Path, default=DEFAULT_ALL_TEST_MANIFEST)
    parser.add_argument("--image-dir", required=True,
                         help="Base dir to fall back to for locating a case's image file if it "
                              "isn't in --all-test-manifest, e.g. .../inhouse_abdominal_ct/nifti_resampled")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_excel(args.worksheet, sheet_name="Sheet1", header=1)
    cols = {_normalize_col(c): c for c in df.columns}

    def col(*keywords: str) -> str:
        # Exact match first: a substring-only search on a single keyword like
        # "sentence" would also match "Sentence Mismatch (Y/N)" (which comes
        # first in column order and is blank for most rows), silently grabbing
        # the wrong column instead of the real "Sentence" text column.
        if len(keywords) == 1 and keywords[0] in cols:
            return cols[keywords[0]]
        for norm, orig in cols.items():
            if all(k in norm for k in keywords):
                return orig
        raise KeyError(f"No column matching {keywords} in {list(df.columns)}")

    disposition_col = _find_disposition_column(df)
    accession_col = col("accession")
    series_col = col("series")
    slice_col = col("slice")
    annoidx_col = col("annoidx")
    sentence_col = col("sentence")

    with open(args.all_test_manifest) as f:
        all_test = json.load(f)
    by_mask = {s["mask"]: s for s in all_test}

    image_dir = Path(args.image_dir)
    n_not_good = n_missing_fields = n_missing_image = 0
    entries = []

    for _, row in df.iterrows():
        if str(row.get(disposition_col)).strip().lower() != "good":
            n_not_good += 1
            continue

        accession = row.get(accession_col)
        series, slc, anno_idx, sentence = (row.get(series_col), row.get(slice_col),
                                            row.get(annoidx_col), row.get(sentence_col))
        if any(pd.isna(v) for v in (accession, series, slc, anno_idx, sentence)):
            logger.warning(f"Skipping row with 'good' disposition but missing field(s): accession={accession}")
            n_missing_fields += 1
            continue

        mask_rel = f"{accession}/mask_{int(series)}_{int(slc)}_{int(anno_idx)}.nii.gz"
        existing = by_mask.get(mask_rel)
        if existing is not None:
            image_rel = existing["image"]
            region = existing.get("region")
            finding = existing.get("finding")
        else:
            found = _find_series_file(image_dir / str(accession), int(series))
            if found is None:
                logger.warning(f"Skipping {mask_rel}: no image found for series {int(series)} "
                                f"under {image_dir / str(accession)}")
                n_missing_image += 1
                continue
            image_rel = f"{accession}/{found}"
            region = None
            finding = None

        entries.append({
            "image": image_rel,
            "mask": mask_rel,
            "sentence": str(sentence),
            "region": region,
            "finding": finding,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(entries, f, indent=4)

    logger.info(f"{len(df)} worksheet row(s): {len(entries)} kept, {n_not_good} not marked 'good', "
                f"{n_missing_fields} 'good' but missing field(s), {n_missing_image} 'good' but no image found")
    logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
