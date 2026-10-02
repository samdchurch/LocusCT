#!/usr/bin/env python3
"""
For every sample in the given manifest JSON files (each a list of {"image",
"mask", "sentence", "region", "finding"} dicts), looks up the source
radiology report by exam ID (the first path component of "image", e.g.
"CASE0000000") in report_extraction/output/{ED,OED,Onc}/{exam_id}.json,
and checks whether "sentence" matches one of that report's
abnormal_findings[*].phrase entries:

  - Exact match: adds an "in-context" key with that finding's surrounding
    report text. "sentence" is left unchanged.
  - "sentence" is a strict substring of a finding's phrase (but not equal to
    it): overwrites "sentence" with the finding's full phrase, and adds
    "in-context" as above.
  - No match, or no report found for the exam: sample is left untouched.

Manifests are rewritten in place. ReXGroundingCT_*.json and
rexgroundingct_manifest_missing.json are not included in the default list --
they use an unrelated external ID scheme with no matching report.

Usage:
    python add_in_context_findings.py
    python add_in_context_findings.py --manifests official_splits/curated_ed_val_data.json
"""
import argparse
import json
import re
from pathlib import Path

GROUNDER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_SPLITS = GROUNDER_ROOT / "official_splits"
DEFAULT_REPORT_DIR = GROUNDER_ROOT.parent / "report_extraction" / "output"

DEFAULT_MANIFESTS = [
    DEFAULT_OFFICIAL_SPLITS / name
    for name in (
        "all_data_train.json",
        "all_data_train_10pct.json",
        "all_data_train_30pct.json",
        "all_data_val.json",
        "all_test_data.json",
        "all_train_val_data.json",
        "curated_ed_onc_val_data.json",
        "curated_ed_train_data.json",
        "curated_ed_train_val_data.json",
        "curated_ed_val_data.json",
        "curated_onc_train_data.json",
        "curated_onc_train_val_data.json",
        "curated_onc_val_data.json",
        "ed_official_test_data.json",
        "onc_official_test_data.json",
    )
]

EXAM_ID_RE = re.compile(r"^(ED|OED|Onc)Detect\d{6}[a-z]$")


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def cohort_for_exam(exam_id: str) -> str | None:
    m = EXAM_ID_RE.match(exam_id)
    return m.group(1) if m else None


def load_findings(report_dir: Path, cohort: str, exam_id: str, cache: dict) -> list[tuple[str, dict]] | None:
    """Returns [(normalized_phrase, entry), ...] in report order, or None if the report is missing/unreadable."""
    key = (cohort, exam_id)
    if key in cache:
        return cache[key]
    path = report_dir / cohort / f"{exam_id}.json"
    try:
        with open(path) as f:
            report = json.load(f)
    except FileNotFoundError:
        cache[key] = None
        return None
    except json.JSONDecodeError:
        print(f"WARNING: malformed report JSON, treating as no-report: {path}")
        cache[key] = None
        return None
    abnormal = report.get("abnormal_findings") or {}
    findings = [(normalize_ws(entry["phrase"]), entry) for entry in abnormal.values()]
    cache[key] = findings
    return findings


def find_match(norm_sentence: str, findings: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    for norm_phrase, entry in findings:
        if norm_phrase == norm_sentence:
            return "exact", entry

    candidates = [
        (len(norm_phrase), entry)
        for norm_phrase, entry in findings
        if norm_sentence and norm_sentence in norm_phrase and norm_phrase != norm_sentence
    ]
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return "partial", candidates[0][1]

    return None


def process_manifest(path: Path, report_dir: Path, cache: dict) -> dict[str, int]:
    with open(path) as f:
        samples = json.load(f)
    counts = dict(total=len(samples), exact=0, partial=0, no_report=0, unmatched=0)

    for sample in samples:
        image = sample.get("image") or ""
        exam_id = image.split("/")[0] if image else ""
        cohort = cohort_for_exam(exam_id)
        if cohort is None:
            counts["no_report"] += 1
            continue

        findings = load_findings(report_dir, cohort, exam_id, cache)
        if findings is None:
            counts["no_report"] += 1
            continue

        match = find_match(normalize_ws(sample.get("sentence") or ""), findings)
        if match is None:
            counts["unmatched"] += 1
            continue

        kind, entry = match
        if kind == "partial":
            sample["sentence"] = entry["phrase"]
            counts["partial"] += 1
        else:
            counts["exact"] += 1
        sample["in-context"] = entry["in-context"]

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(samples, f, indent=4)
    tmp.replace(path)

    print(
        f"{path.name}: {counts['total']} samples -- {counts['exact']} exact, "
        f"{counts['partial']} partial/updated, {counts['no_report']} no-report, "
        f"{counts['unmatched']} unmatched"
    )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", nargs="+", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache: dict = {}
    totals = dict(total=0, exact=0, partial=0, no_report=0, unmatched=0)

    for path in args.manifests:
        counts = process_manifest(path, args.report_dir, cache)
        for k in totals:
            totals[k] += counts[k]

    print("-" * 60)
    print(
        f"TOTAL: {totals['total']} samples -- {totals['exact']} exact, "
        f"{totals['partial']} partial/updated, {totals['no_report']} no-report, "
        f"{totals['unmatched']} unmatched"
    )


if __name__ == "__main__":
    main()
