#!/usr/bin/env python3
"""
Resample the Merlin abdominal CT dataset's full-volume CTs to the same
(2.0 x 2.0 x 3.0 mm) voxel spacing and (192 x 192 x 192) voxel shape used for
the Grounder training images' 192^3 grid (see resample_and_prune.py /
resample_and_crop.py's --target-spacing 2.0 2.0 3.0 --target-shape 192 192 192,
which this mirrors -- same resample -> centre-of-mass crop/pad -> affine
pipeline, just pointed at a different input/output root and collecting files
recursively rather than assuming one level of <accession> subfolders, since
Merlin's own directory layout under merlin_data isn't necessarily the same
shape).

Output files already existing at the destination are skipped (see
process_file), so re-running after an interruption does NOT reprocess
already-resampled files -- delete the stale ones (or their containing
directory) first if you need them regenerated.

Input  : /path/to/data/public_datasets/merlinabdominalctdataset/merlin_data/**/*.nii.gz  (default -- see --input-root)
Output : /path/to/data/public_datasets/merlinabdominalctdataset/merlin_data_resampled_192/**/*.nii.gz  (default -- see --output-root)
         (mirrors each input file's path relative to --input-root)

--input-root/--output-root/--target-spacing/--target-shape default to the
values above, so any existing invocation with no new flags is unaffected.
Pass all four together to produce an alternate resampling from a different
source tree into its own output folder -- e.g. VoxTell --from-scratch's own
1.5x1.5x3.0mm/352x352x180 grid (see resample_and_crop.py's own
DEFAULT_TARGET_SPACING/DEFAULT_TARGET_SHAPE) instead of this file's default
192^3 grid, without touching the default merlin_data_resampled_192/ output:

    python resample_merlin.py \\
        --input-root /path/to/data/merlinabdominalctdataset/merlin_data/ICLR2027_merlin_images \\
        --output-root /path/to/data/merlinabdominalctdataset/merlin_data/ICLR2027_merlin_images_resampled_352x352x180 \\
        --target-spacing 1.5 1.5 3.0 --target-shape 352 352 180

Usage
-----
    python resample_merlin.py              # process everything
    python resample_merlin.py --workers 8  # parallelise over N cores
    python resample_merlin.py --dry-run    # print what would be done
"""

import os

# Work around a known OpenBLAS CPU-dispatch bug on some AMD Zen4 EPYC nodes, where
# auto-detection misidentifies the CPU and selects an Intel-only AVX-512 kernel,
# crashing with SIGILL. Must be set before numpy/scipy are imported.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

import argparse
import json
import logging
import multiprocessing as mp
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# ── paths (all overridable -- see --input-root/--output-root/--target-spacing/
#    --target-shape) ──────────────────────────────────────────────────────────
DEFAULT_INPUT_ROOT  = Path("/path/to/data/public_datasets/merlinabdominalctdataset/merlin_data")
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/public_datasets/merlinabdominalctdataset/merlin_data_resampled_192")

# ── target parameters (default: must match resample_and_prune.py's GRID_192_
#    SPACING / GRID_192_SHAPE -- these are the 192^3 grid Grounder's trained/
#    fine-tuned checkpoints expect. Pass --target-spacing/--target-shape
#    together with --input-root/--output-root for an alternate resampling
#    into its own output folder, e.g. VoxTell --from-scratch's own
#    1.5x1.5x3.0mm/352x352x180 grid -- same convention as resample_and_crop.py's
#    own --target-spacing/--target-shape/--output-root) ────────────────────────
DEFAULT_TARGET_SPACING = (2.0, 2.0, 3.0)   # mm  (x, y, z)
DEFAULT_TARGET_SHAPE   = (192, 192, 192)   # voxels (x, y, z)
AIR_HU = -1000.0  # fill value for image padding (not 0 -- see crop_pad_to_shape)

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
def get_voxel_spacing(img: nib.Nifti1Image) -> np.ndarray:
    """Return voxel sizes (x, y, z) in mm from the image header."""
    zooms = np.abs(img.header.get_zooms()[:3])
    if np.any(zooms == 0):
        raise ValueError(f"Zero voxel spacing detected: {zooms}")
    return zooms.astype(float)


def resample_volume(
    data: np.ndarray,
    current_spacing: np.ndarray,
    target_spacing: np.ndarray,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Zoom `data` so that its physical voxel size matches `target_spacing`.
    Returns (resampled_data, actual_output_spacing).
    """
    zoom_factors = current_spacing / target_spacing
    resampled = zoom(data, zoom_factors, order=order, mode="nearest", prefilter=True)
    # actual spacing may differ slightly from target due to integer rounding
    actual_spacing = current_spacing / zoom_factors
    return resampled, actual_spacing


def centre_of_mass_nonzero(data: np.ndarray, air_threshold: float = -900.0) -> np.ndarray:
    """
    Return the voxel index of the centre of mass of tissue voxels (HU >
    air_threshold), excluding background air and out-of-FOV padding.

    Real HU data is almost never exactly 0 (air is -1000, not 0), so a plain
    `data != 0` check would count virtually every voxel as "non-zero" --
    degenerating to the plain geometric centre of the array regardless of
    where the patient actually is. -900 sits just above true air/padding
    (-1000 and below) while still including low-density aerated lung.
    """
    mask = data > air_threshold
    if not mask.any():
        # fall back to geometric centre
        return np.array(data.shape) / 2.0
    coords = np.argwhere(mask).astype(float)
    return coords.mean(axis=0)


def crop_pad_to_shape(
    data: np.ndarray,
    target_shape: tuple[int, int, int],
    centre: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Crop or pad `data` so the output has exactly `target_shape`, keeping
    `centre` (fractional voxel index) as close to the output centre as
    possible. Padded regions are filled with `fill_value`.

    Pass an air-equivalent HU value (AIR_HU) for images -- the default 0.0
    reads as water-density, not background, and would inject a block of fake
    mid-range tissue values into any volume needing padding (common, since
    TARGET_SHAPE's physical FOV often exceeds a scan's native coverage).
    """
    out = np.full(target_shape, fill_value, dtype=data.dtype)
    centre = np.round(centre).astype(int)

    for dim in range(3):
        half = target_shape[dim] // 2

        # desired slice in the source volume
        src_start = centre[dim] - half
        src_end   = src_start + target_shape[dim]

        # clamp to valid source range
        src_start_clamped = max(src_start, 0)
        src_end_clamped   = min(src_end, data.shape[dim])

        # corresponding slice in the output array
        dst_start = src_start_clamped - src_start
        dst_end   = dst_start + (src_end_clamped - src_start_clamped)

        # store as slice objects for clean indexing below
        if dim == 0:
            s0 = (slice(src_start_clamped, src_end_clamped), slice(None), slice(None))
            d0 = (slice(dst_start, dst_end),                 slice(None), slice(None))
        elif dim == 1:
            s1 = (slice(None), slice(src_start_clamped, src_end_clamped), slice(None))
            d1 = (slice(None), slice(dst_start, dst_end),                 slice(None))
        else:
            s2 = (slice(None), slice(None), slice(src_start_clamped, src_end_clamped))
            d2 = (slice(None), slice(None), slice(dst_start, dst_end))

    # apply all three dimensions in one pass
    src_slices = (s0[0], s1[1], s2[2])
    dst_slices = (d0[0], d1[1], d2[2])
    out[dst_slices] = data[src_slices]
    return out


def build_affine(
    original_affine: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: np.ndarray,
    resampled_shape: np.ndarray,
    final_shape: tuple[int, int, int],
    com_resampled: np.ndarray,
) -> np.ndarray:
    """
    Construct the NIfTI affine for the cropped/padded output volume so that
    spatial coordinates are preserved correctly.
    """
    # New voxel sizes
    scale = target_spacing / original_spacing

    # Direction cosines from original affine (columns 0-2, normalised)
    directions = original_affine[:3, :3] / original_spacing  # unit vectors * old spacing / old spacing

    # Build rotation/scale part
    new_affine = np.eye(4)
    new_affine[:3, :3] = directions * target_spacing

    # Origin: the world coordinate of the first voxel in the cropped output.
    # com_resampled is the CoM in the resampled volume.
    half = np.array(final_shape) // 2
    first_voxel_in_resampled = com_resampled - half  # may be negative (pad case)

    # Convert that resampled-volume voxel index to world coords using the
    # resampled-volume affine (same directions, new spacing, same origin as original).
    resampled_affine = np.eye(4)
    resampled_affine[:3, :3] = directions * target_spacing
    resampled_affine[:3,  3] = original_affine[:3, 3]

    origin_world = resampled_affine[:3, :3] @ first_voxel_in_resampled + resampled_affine[:3, 3]
    new_affine[:3, 3] = origin_world
    return new_affine


# ──────────────────────────────────────────────────────────────────────────────
def process_file(
    src: Path,
    dst: Path,
    target_spacing: np.ndarray,
    target_shape: tuple[int, int, int],
    dry_run: bool = False,
) -> str:
    """
    Full pipeline for a single NIfTI file.
    Returns a short status string.
    """
    if dst.exists():
        return f"SKIP  (already exists)  {dst.name}"

    if dry_run:
        return f"DRY   {src}  ->  {dst}"

    try:
        img  = nib.load(str(src))
        data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        return f"CORRUPT  {src.parent.name}/{src.name}  ({e})"

    orig_spacing = get_voxel_spacing(img)

    log.info("START  %s/%s  shape=%s  spacing=%s  size=%.0fMB",
              src.parent.name, src.name, list(data.shape), np.round(orig_spacing, 2),
              data.nbytes / 1e6)

    # ── 1. resample ────────────────────────────────────────────────────────────
    resampled_data, actual_spacing = resample_volume(data, orig_spacing, target_spacing)

    # ── 2. centre of mass on resampled volume ──────────────────────────────────
    com = centre_of_mass_nonzero(resampled_data)

    # ── 3. crop / pad ──────────────────────────────────────────────────────────
    final_data = crop_pad_to_shape(resampled_data, target_shape, com, fill_value=AIR_HU)

    # ── 4. build correct affine ────────────────────────────────────────────────
    new_affine = build_affine(
        original_affine  = img.affine,
        original_spacing = orig_spacing,
        target_spacing   = target_spacing,
        resampled_shape  = np.array(resampled_data.shape),
        final_shape      = target_shape,
        com_resampled    = com,
    )

    # ── 5. save ────────────────────────────────────────────────────────────────
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(final_data, new_affine, header=img.header)
    out_img.header.set_zooms(target_spacing)
    out_img.header.set_data_shape(target_shape)
    nib.save(out_img, str(dst))

    return f"OK    {src.parent.name}/{src.name}  {list(data.shape)}@{np.round(orig_spacing,2)}  ->  {list(final_data.shape)}@{target_spacing}"


def _isolated_target(
    q: "mp.Queue", src: Path, dst: Path, target_spacing: np.ndarray, target_shape: tuple[int, int, int],
) -> None:
    try:
        q.put(("ok", process_file(src, dst, target_spacing, target_shape)))
    except Exception:
        q.put(("error", traceback.format_exc()))


def run_isolated(
    src: Path, dst: Path, target_spacing: np.ndarray, target_shape: tuple[int, int, int],
) -> str:
    """
    Run `process_file` for a single file in its own fresh subprocess, so that if it
    crashes we can read the child's actual exit signal (SIGKILL from an OOM kill,
    SIGSEGV/SIGABRT/SIGBUS from a native crash in scipy/nibabel, etc.) instead of the
    generic BrokenProcessPool message a shared pool would give.
    """
    ctx = mp.get_context()
    result_q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_isolated_target, args=(result_q, src, dst, target_spacing, target_shape))
    proc.start()
    proc.join()

    if not result_q.empty():
        status, payload = result_q.get()
        if status == "ok":
            return payload
        return f"FAIL  {src.parent.name}/{src.name}\n{payload}"

    exitcode = proc.exitcode
    if exitcode is not None and exitcode < 0:
        sig = -exitcode
        try:
            sig_name = signal.Signals(sig).name
        except ValueError:
            sig_name = f"signal {sig}"
        hint = {
            "SIGKILL": "likely OOM-killed by the OS",
            "SIGSEGV": "native segfault in scipy/nibabel, not OOM",
            "SIGABRT": "native abort (e.g. malloc corruption or C assertion), not OOM",
            "SIGBUS":  "bus error, often truncated/corrupt input or a filesystem I/O issue, not OOM",
            "SIGFPE":  "floating point exception in native code, not OOM",
            "SIGILL":  "illegal CPU instruction, not OOM — likely a numpy/scipy build using "
                       "instructions (e.g. AVX-512) this node's CPU doesn't support",
        }.get(sig_name, "unexpected termination signal")
        reason = f"killed by {sig_name} ({hint})"
    else:
        reason = f"exited with code {exitcode} and produced no result"

    return f"CRASH  {src.parent.name}/{src.name}  ({reason})"


def collect_jobs(input_root: Path, output_root: Path, subdir: str | None = None) -> list[tuple[Path, Path]]:
    """
    Collect (src, dst) pairs by recursively finding every *.nii.gz under
    input_root (Merlin's own on-disk layout isn't assumed to be one level of
    per-accession subfolders like the main in-house dataset -- rglob finds files at
    whatever depth they're actually at) and mirroring each file's path
    relative to input_root under output_root.

    If `subdir` is given, only recurse under input_root/subdir (for splitting
    work across Slurm array tasks by top-level subfolder).
    """
    root = input_root / subdir if subdir else input_root
    if not root.is_dir():
        log.error("Input directory not found: %s", root)
        return []

    jobs = []
    for src in sorted(root.rglob("*.nii.gz")):
        rel = src.relative_to(input_root)
        dst = output_root / rel
        jobs.append((src, dst))
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    parser.add_argument("--subdir", type=str, default=None,
                        help="Process only this subfolder (relative to --input-root), for splitting across Slurm array tasks")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                        help=f"Input directory of source NIfTI files (default: {DEFAULT_INPUT_ROOT})")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--target-spacing", type=float, nargs=3, default=list(DEFAULT_TARGET_SPACING),
                        metavar=("X", "Y", "Z"), help=f"Target voxel spacing in mm (default: {DEFAULT_TARGET_SPACING})")
    parser.add_argument("--target-shape", type=int, nargs=3, default=list(DEFAULT_TARGET_SHAPE),
                        metavar=("X", "Y", "Z"), help=f"Target crop/pad shape in voxels (default: {DEFAULT_TARGET_SHAPE})")
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root
    target_spacing = np.array(args.target_spacing)
    target_shape = tuple(args.target_shape)

    jobs = collect_jobs(input_root, output_root, subdir=args.subdir)
    log.info("Found %d NIfTI file(s) under %s", len(jobs), input_root / args.subdir if args.subdir else input_root)

    if not jobs:
        log.error("No files found under %s", input_root)
        sys.exit(1)

    ok = skip = corrupt = fail = 0
    failures: list[dict] = []

    def _record(src: Path, msg: str) -> None:
        nonlocal ok, skip, corrupt, fail
        if msg.startswith("SKIP"):
            skip += 1
        elif msg.startswith("CORRUPT"):
            corrupt += 1
            failures.append({"file": str(src), "status": "corrupt", "reason": msg})
        else:
            ok += 1

    if args.workers == 1 or args.dry_run:
        for src, dst in tqdm(jobs, desc="Resampling"):
            try:
                msg = process_file(src, dst, target_spacing, target_shape, dry_run=args.dry_run)
                level = logging.WARNING if msg.startswith("CORRUPT") else logging.INFO
                log.log(level, msg)
                _record(src, msg)
            except Exception as e:
                log.error("FAIL  %s\n%s", src, traceback.format_exc())
                fail += 1
                failures.append({"file": str(src), "status": "failed", "reason": str(e)})
    else:
        def _finish(src: Path, get_result) -> bool:
            """Record the outcome of a finished future. Returns True if it failed only
            because the shared worker pool was poisoned by some *other* file crashing
            it (BrokenProcessPool), meaning this file was never actually attempted and
            deserves a real retry rather than being logged as a failure."""
            nonlocal fail
            try:
                msg = get_result()
            except BrokenProcessPool:
                return True
            except Exception as e:
                log.error("FAIL  %s\n%s", src, traceback.format_exc())
                fail += 1
                failures.append({"file": str(src), "status": "failed", "reason": str(e)})
                return False
            level = logging.WARNING if msg.startswith("CORRUPT") else logging.INFO
            log.log(level, msg)
            _record(src, msg)
            return False

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_file, src, dst, target_spacing, target_shape): (src, dst) for src, dst in jobs}
            to_retry = []
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Resampling"):
                src, dst = futures[fut]
                if _finish(src, fut.result):
                    to_retry.append((src, dst))

        if to_retry:
            # A worker dying abruptly poisons the whole pool: every other still-pending
            # future then also raises BrokenProcessPool, even though most of those files
            # were never actually attempted. Retry each one individually via run_isolated,
            # which reports the real exit signal if it crashes again on its own; otherwise
            # it was just collateral damage and now gets processed properly.
            log.warning("Worker pool crashed; retrying %d file(s) individually to separate "
                        "genuine crashes from collateral pool poisoning", len(to_retry))
            for src, dst in tqdm(to_retry, desc="Retrying"):
                msg = run_isolated(src, dst, target_spacing, target_shape)
                if msg.startswith(("CRASH", "FAIL")):
                    log.error(msg)
                    fail += 1
                    failures.append({"file": str(src), "status": "crashed", "reason": msg})
                else:
                    level = logging.WARNING if msg.startswith("CORRUPT") else logging.INFO
                    log.log(level, msg)
                    _record(src, msg)

    log.info("─" * 60)
    log.info("Done.  OK=%d  SKIPPED=%d  CORRUPT=%d  FAILED=%d", ok, skip, corrupt, fail)

    if failures and not args.dry_run:
        suffix = f"_{args.subdir.replace('/', '_')}" if args.subdir else ""
        failures_path = output_root / f"failures{suffix}.json"
        output_root.mkdir(parents=True, exist_ok=True)
        with open(failures_path, "w") as f:
            json.dump(failures, f, indent=2)
        log.info("Wrote %d failed example(s) to %s", len(failures), failures_path)

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
