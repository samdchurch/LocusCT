#!/usr/bin/env python3
"""
Per-file pipeline: resamples each native CT image to BOTH of our standard
grids, then deletes the native original immediately if it isn't referenced
by any train/val/test manifest. Interleaved per file (not two separate
passes over the whole dataset), so disk usage never needs to hold the full
native dataset AND both full resampled grids at once -- unlike
submit_resample_migrate_and_192.sh + submit_delete_unreferenced_originals.sh,
which assumed enough headroom for that. Deleting a native original only
ever happens after both of its resamples for that file succeeded (or were
already done) -- never on a resample failure.

Grids (hardcoded -- this script is specifically for these two, not a
generic multi-grid tool; see resample_and_crop.py for a general
one-grid-at-a-time tool):
  nifti_resampled/      1.5x1.5x3.0mm, 352x352x192  (--force: regenerates
                         existing files there, which are the OLD
                         352x352x180 shape)
  nifti_resampled_192/  2.0x2.0x3.0mm, 192^3         (skip-if-exists)

Reuses resample_and_crop.py's process_file() (imported directly, called
once per grid per file) and check_unused_nifti_files.py's referenced_paths()
for manifest membership -- no resampling or reference-checking logic is
duplicated here.

Deletion is irreversible. --dry-run skips both resampling and deletion,
just reporting what WOULD happen -- use it (optionally with --accession or
--limit for a quick subset) to sanity-check before a real run. For a real
run, --confirm-delete is required or nothing gets deleted (files still get
resampled either way, without freeing any space).

Usage
-----
    python resample_and_prune.py --dry-run                                   # report only
    python resample_and_prune.py --accession CASE0000000 --confirm-delete  # validate on one accession
    python resample_and_prune.py --workers 16 --confirm-delete               # real run
"""

import argparse
import json
import logging
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from check_unused_nifti_files import DEFAULT_MANIFESTS, DEFAULT_NIFTI_DIR, referenced_paths
from resample_and_crop import process_file as resample_process_file

GRID_352_SPACING = np.array([1.5, 1.5, 3.0])
GRID_352_SHAPE = (352, 352, 192)
GRID_352_OUTPUT_ROOT = Path("/path/to/data/inhouse_abdominal_ct/nifti_resampled")

GRID_192_SPACING = np.array([2.0, 2.0, 3.0])
GRID_192_SHAPE = (192, 192, 192)
GRID_192_OUTPUT_ROOT = Path("/path/to/data/inhouse_abdominal_ct/nifti_resampled_192")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def process_one(
    src: Path, rel_path: str, referenced: bool,
    grid_352_root: Path, grid_192_root: Path,
    dry_run: bool, confirm_delete: bool,
) -> dict:
    """Resample one native file to both grids, then decide the original's fate."""
    msg_352 = resample_process_file(
        src, grid_352_root / rel_path, GRID_352_SPACING, GRID_352_SHAPE, dry_run=dry_run, force=True,
    )
    msg_192 = resample_process_file(
        src, grid_192_root / rel_path, GRID_192_SPACING, GRID_192_SHAPE, dry_run=dry_run, force=False,
    )
    resample_ok = msg_352.startswith(("OK", "SKIP", "DRY")) and msg_192.startswith(("OK", "SKIP", "DRY"))

    if referenced:
        action = "kept_referenced"
    elif not resample_ok:
        action = "kept_resample_failed"
    elif not dry_run and confirm_delete:
        src.unlink()
        action = "deleted"
    else:
        action = "would_delete"

    return {
        "path": rel_path, "referenced": referenced,
        "grid_352": msg_352, "grid_192": msg_192,
        "action": action, "time": datetime.now(timezone.utc).isoformat(),
    }


def collect_native_files(nifti_dir: Path, accession: str | None) -> list[tuple[Path, str]]:
    candidates = [nifti_dir / accession] if accession else sorted(nifti_dir.iterdir())
    jobs = []
    for acc_dir in candidates:
        if not acc_dir.is_dir():
            log.warning("Accession directory not found: %s", acc_dir)
            continue
        for src in sorted(acc_dir.glob("*.nii.gz")):
            rel_path = str(src.relative_to(nifti_dir)).replace("\\", "/")
            jobs.append((src, rel_path))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nifti-dir", default=DEFAULT_NIFTI_DIR,
                         help=f"Native NIfTI directory (default: {DEFAULT_NIFTI_DIR})")
    parser.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS,
                         help=f"Manifests to check reference membership against (default: {DEFAULT_MANIFESTS})")
    parser.add_argument("--grid-352-root", type=Path, default=GRID_352_OUTPUT_ROOT)
    parser.add_argument("--grid-192-root", type=Path, default=GRID_192_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                         help="Report only -- no resampling, no deletion")
    parser.add_argument("--confirm-delete", action="store_true",
                         help="Actually delete unreferenced originals after they're resampled "
                              "(default: resample only, log what WOULD be deleted -- does not free space)")
    parser.add_argument("--accession", default=None,
                         help="Process only this accession folder (for validating on a subset)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N files found (for a quick smoke test)")
    parser.add_argument("--audit-log", default="resample_and_prune_log.jsonl",
                         help="Path to write one JSON line per processed file (default: resample_and_prune_log.jsonl)")
    args = parser.parse_args()

    nifti_dir = Path(args.nifti_dir)
    jobs = collect_native_files(nifti_dir, args.accession)
    if args.limit:
        jobs = jobs[:args.limit]
    log.info("Found %d native file(s)", len(jobs))

    referenced = referenced_paths(args.manifests, field="image")
    log.info("Referenced across %d manifest(s): %d unique path(s)", len(args.manifests), len(referenced))

    counts: Counter = Counter()
    with open(args.audit_log, "w") as log_f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one, src, rel_path, rel_path in referenced,
                args.grid_352_root, args.grid_192_root, args.dry_run, args.confirm_delete,
            ): rel_path
            for src, rel_path in jobs
        }
        for fut in as_completed(futures):
            rel_path = futures[fut]
            try:
                result = fut.result()
            except Exception:
                log.error("FAIL  %s\n%s", rel_path, traceback.format_exc())
                result = {"path": rel_path, "action": "FAIL", "time": datetime.now(timezone.utc).isoformat()}

            log_f.write(json.dumps(result) + "\n")
            log_f.flush()
            counts[result["action"]] += 1
            if result["action"] not in ("kept_referenced",):
                log.info("%s  %s", result["action"], rel_path)

    log.info("-" * 60)
    log.info("Done. %s", dict(counts))


if __name__ == "__main__":
    main()
