#!/usr/bin/env python3
"""
Deletes native NIfTI originals under --nifti-dir that aren't referenced by
any of --manifests' "image" entries -- reuses check_unused_nifti_files.py's
referenced_paths() for the membership check, so nothing is duplicated.

Only ever touches files under --nifti-dir (the native originals); it never
touches resampled folders (nifti_resampled/, nifti_resampled_192/, etc.).

Deletion is irreversible, so this is dry-run by default: it writes what it
WOULD delete to --audit-log (as one JSON object) without removing anything.
Pass --confirm-delete to actually delete -- in that mode --audit-log is
instead one JSON object per line (same convention as finetune_voxtell.py's
metrics.jsonl), appended and flushed immediately after each file is removed,
so a mid-run crash still leaves a valid, accurate record of what was
actually deleted.

Usage
-----
    python delete_unreferenced_originals.py                    # dry run
    python delete_unreferenced_originals.py --confirm-delete    # actually delete
    python delete_unreferenced_originals.py --nifti-dir /path/to/nifti --audit-log audit.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from check_unused_nifti_files import DEFAULT_MANIFESTS, DEFAULT_NIFTI_DIR, referenced_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nifti-dir", default=DEFAULT_NIFTI_DIR,
                         help=f"Native NIfTI directory to prune (default: {DEFAULT_NIFTI_DIR})")
    parser.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS,
                         help=f"Manifest JSON files to check against (default: {DEFAULT_MANIFESTS})")
    parser.add_argument("--confirm-delete", action="store_true",
                         help="Actually delete unreferenced files (default: dry run only)")
    parser.add_argument("--audit-log", default="deleted_nifti_originals.json",
                         help="Path to write the dry-run/deletion audit log (default: deleted_nifti_originals.json)")
    args = parser.parse_args()

    nifti_dir = Path(args.nifti_dir)
    on_disk = {
        str(p.relative_to(nifti_dir)).replace("\\", "/")
        for p in nifti_dir.glob("*/*.nii.gz")
    }
    referenced = referenced_paths(args.manifests, field="image")
    unreferenced = sorted(on_disk - referenced)

    print(f"On disk under {nifti_dir}: {len(on_disk)} file(s)")
    print(f"Referenced across {len(args.manifests)} manifest(s): {len(referenced)} unique path(s)")
    print(f"Unreferenced: {len(unreferenced)}")

    if not args.confirm_delete:
        with open(args.audit_log, "w") as f:
            json.dump({"dry_run": True, "count": len(unreferenced), "would_delete": unreferenced}, f, indent=2)
        print(f"DRY RUN -- wrote {len(unreferenced)} candidate path(s) to {args.audit_log}. "
              f"Pass --confirm-delete to actually remove them.")
        return

    deleted = []
    with open(args.audit_log, "w") as f:
        for rel_path in unreferenced:
            path = nifti_dir / rel_path
            path.unlink()
            deleted.append(rel_path)
            record = {"path": rel_path, "deleted_at": datetime.now(timezone.utc).isoformat()}
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"Deleted {len(deleted)} file(s). Audit log (one JSON object per line): {args.audit_log}")


if __name__ == "__main__":
    main()
