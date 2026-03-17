#!/usr/bin/env python3
"""
run_gsp_pipeline.py
===================
End-to-end orchestrator for the GSP phantom QC pipeline.

Stages
------
  Stage 1  DWI processing  (pydra_dwi_processing.py)
           Runs if DWI acquisitions are found in --input-dir.
           Outputs per DWI series:
             T1_n4_in_DWI_space.nii.gz, ADC.nii.gz, FA.nii.gz,
             DWI_preproc_biascorr.mif.gz

  Stage 2  Phantom QC in DWI space  (pydra_phantom_iterative.py)
           Runs once per DWI series produced by Stage 1.
           Input image: T1_n4_in_DWI_space.nii.gz from Stage 1.
           The ADC and FA files in the same folder are picked up
           automatically by the phantom processor, so per-vial
           ADC / FA metrics are extracted and plotted.

  Stage 3  Phantom QC on native contrasts  (pydra_phantom_iterative.py)
           Runs if the input directory contains IR and/or TE series
           (folders matching 'se_ir' or 't2_se' / 'te', case-insensitive).
           All contrast DICOMs (T1, IR, TE) are converted to NIfTI into a
           temporary staging folder; the phantom processor then registers
           the T1 and extracts vial metrics from every NIfTI present,
           producing T1, T2 parametric maps and per-contrast scatter plots.

Usage
-----
  python run_gsp_pipeline.py \\
    --input-dir  /path/to/patient/scans \\
    --output-dir /path/to/outputs \\
    --phantom    SPIRIT

  Optional flags:
    --denoise-degibbs   pass to DWI pipeline
    --gradcheck         pass to DWI pipeline
    --nocleanup         keep DWI tmp/ directories
    --readout-time      override TotalReadoutTime (seconds)
    --eddy-options      override FSL eddy options string
    --dry-run           plan and print; do not execute

Notes
-----
  - TemplateData/ and rotations.txt are resolved relative to this script's
    location.  Expected layout:
      <repo>/run_gsp_pipeline.py          ← this file
      <repo>/Functions/pydra_dwi_processing.py
      <repo>/Functions/pydra_phantom_iterative.py
      <repo>/TemplateData/<phantom>/ImageTemplate.nii.gz
      <repo>/TemplateData/<phantom>/VialsLabelled/
      <repo>/TemplateData/rotations.txt
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path


# ---------------------------------------------------------------------------
# Path constants (relative to this script)
#
# The script may live either at the repo root OR inside Functions/.
# We resolve REPO_ROOT by walking up until we find TemplateData/.
# ---------------------------------------------------------------------------


def _find_repo_root() -> Path:
    """
    Walk upward from this script's location until a directory containing
    both 'TemplateData' and 'Functions' is found.  Raises RuntimeError if
    not found within 3 levels.
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(3):
        if (candidate / "TemplateData").is_dir() and (candidate / "Functions").is_dir():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(
        f"Could not locate repo root (expected a directory containing both "
        f"'TemplateData/' and 'Functions/') within 3 levels of {Path(__file__).resolve()}"
    )


REPO_ROOT = _find_repo_root()
FUNCTIONS_DIR = REPO_ROOT / "Functions"
TEMPLATE_DATA_ROOT = REPO_ROOT / "TemplateData"
ROTATIONS_FILE = TEMPLATE_DATA_ROOT / "rotations.txt"

DWI_SCRIPT = FUNCTIONS_DIR / "pydra_dwi_processing.py"
PHANTOM_SCRIPT = FUNCTIONS_DIR / "pydra_phantom_iterative.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_header(title: str):
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def run_cmd(cmd: list, label: str):
    """Run a subprocess command, streaming output and raising on failure."""
    print(f"  >> {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode}).")


def convert_dicom_dir(dicom_dir: Path, out_dir: Path) -> list:
    """
    Run dcm2niix on dicom_dir, placing NIfTIs in out_dir.
    Returns list of produced .nii.gz paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["dcm2niix", "-o", str(out_dir), "-f", "%p", "-z", "y", str(dicom_dir)],
        check=True,
        capture_output=True,
    )
    niis = sorted(out_dir.glob("*.nii.gz"))
    if not niis:
        raise FileNotFoundError(f"dcm2niix produced no NIfTI from {dicom_dir}")
    return [str(p) for p in niis]


def scan_input_dir(input_dir: Path) -> dict:
    """
    Classify subdirectories in input_dir into:
      t1_dirs      – folders matching 't1' (case-insensitive)
      ir_dirs      – folders matching 'se_ir' or standalone 'ir'
      te_dirs      – folders matching 't2_se' or standalone 'te'
      dwi_dirs     – folders matching '_diff_' or '_DWI_' (case-insensitive)
                     AND not ending in _ADC or _FA
      other_dirs   – everything else

    Returns a dict with those keys plus 'has_dwi' and 'has_native_contrasts'
    boolean flags for easy branching.
    """
    t1_dirs, ir_dirs, te_dirs, dwi_dirs, other_dirs = [], [], [], [], []

    for d in sorted(input_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        # Skip scanner-derived maps
        if re.search(r"_(ADC|FA)$", name, re.IGNORECASE):
            other_dirs.append(d)
            continue

        if re.search(r"t1", name, re.IGNORECASE):
            t1_dirs.append(d)
        elif re.search(r"se_ir|(?<![a-z0-9])ir(?![a-z0-9])", name, re.IGNORECASE):
            ir_dirs.append(d)
        elif re.search(r"t2_se|(?<![a-z0-9])te(?![a-z0-9])", name, re.IGNORECASE):
            te_dirs.append(d)
        elif re.search(r"(_diff_|_DWI_)", name, re.IGNORECASE):
            dwi_dirs.append(d)
        else:
            other_dirs.append(d)

    has_dwi = bool(dwi_dirs)
    has_native_contrasts = bool(ir_dirs or te_dirs)

    return {
        "t1_dirs": t1_dirs,
        "ir_dirs": ir_dirs,
        "te_dirs": te_dirs,
        "dwi_dirs": dwi_dirs,
        "other_dirs": other_dirs,
        "has_dwi": has_dwi,
        "has_native_contrasts": has_native_contrasts,
    }


def derive_session_name(input_dir: Path) -> str:
    """
    Derive a clean session name from the input directory.
    Strategy: use the input directory's own name, stripping any leading
    series-number prefix (e.g. '87-PatientID_Study' → 'PatientID_Study').
    If the name is a pure number, use the full path's parent name instead.
    """
    name = input_dir.name
    # Strip leading digits-dash prefix
    stripped = re.sub(r"^\d+-", "", name)
    if stripped:
        return stripped
    return name


def find_dwi_output_t1(dwi_output_dir: Path) -> Path | None:
    """
    Find the T1 image in a DWI output directory.
    Returns T1_n4_in_DWI_space.nii.gz (registration performed) or
    T1_n4.nii.gz (rpe_none, no registration), or None if neither exists.
    """
    for name in ("T1_n4_in_DWI_space.nii.gz", "T1_n4.nii.gz"):
        candidate = dwi_output_dir / name
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Stage 1: DWI processing
# ---------------------------------------------------------------------------


def run_stage1(input_dir: Path, output_dir: Path, cfg: dict, dry_run: bool) -> list:
    """
    Run pydra_dwi_processing.py on input_dir.
    Returns list of DWI output subdirectory Paths (one per processed series).
    """
    print_header("STAGE 1 — DWI Processing")

    cmd = [
        sys.executable,
        str(DWI_SCRIPT),
        "--scans-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]

    if cfg.get("denoise_degibbs"):
        cmd.append("--denoise-degibbs")
    if cfg.get("gradcheck"):
        cmd.append("--gradcheck")
    if cfg.get("nocleanup"):
        cmd.append("--nocleanup")
    if cfg.get("readout_time") is not None:
        cmd += ["--readout-time", str(cfg["readout_time"])]
    if cfg.get("eddy_options") is not None:
        cmd += ["--eddy-options", cfg["eddy_options"]]

    if dry_run:
        print("  [DRY RUN] Would execute:")
        print(f"  {' '.join(str(c) for c in cmd)}")
        print()
        return []

    run_cmd(cmd, "Stage 1 (DWI processing)")

    # Discover produced DWI output directories by scanning output_dir for
    # subdirectories that contain either T1_n4_in_DWI_space.nii.gz (when
    # registration was performed) or T1_n4.nii.gz (rpe_none, no registration).
    _t1_names = {"T1_n4_in_DWI_space.nii.gz", "T1_n4.nii.gz"}
    dwi_output_dirs = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir() and any((d / n).exists() for n in _t1_names):
            dwi_output_dirs.append(d)

    print(f"\n  Stage 1 complete. Found {len(dwi_output_dirs)} processed DWI series.")
    for d in dwi_output_dirs:
        print(f"    {d.name}")

    return dwi_output_dirs


# ---------------------------------------------------------------------------
# Stage 2: Phantom QC in DWI space
# ---------------------------------------------------------------------------


def run_stage2(
    dwi_output_dirs: list,
    output_dir: Path,
    template_dir: Path,
    dry_run: bool,
):
    """
    For each DWI output directory, run the phantom processor on
    T1_n4_in_DWI_space.nii.gz.  ADC.nii.gz and FA.nii.gz in the same
    folder are picked up automatically.
    """
    print_header("STAGE 2 — Phantom QC in DWI Space")

    if not dwi_output_dirs:
        print("  No DWI output directories to process — skipping Stage 2.\n")
        return

    for dwi_dir in dwi_output_dirs:
        # The T1 filename depends on whether registration was performed:
        #   T1_n4_in_DWI_space.nii.gz — registration performed (rpe_pair/all)
        #   T1_n4.nii.gz              — registration skipped (rpe_none)
        t1_in_dwi = next(
            (
                dwi_dir / n
                for n in ("T1_n4_in_DWI_space.nii.gz", "T1_n4.nii.gz")
                if (dwi_dir / n).exists()
            ),
            None,
        )

        if t1_in_dwi is None:
            print(f"  WARNING: No T1 image found in {dwi_dir.name} — skipping.")
            continue

        print(f"  Processing: {dwi_dir.name}")
        print(f"    Input image: {t1_in_dwi}")
        print(f"    Contrast images found alongside T1:")
        for nii in sorted(dwi_dir.glob("*.nii.gz")):
            print(f"      {nii.name}")

        cmd = [
            sys.executable,
            str(PHANTOM_SCRIPT),
            "single",
            str(t1_in_dwi),
            "--template-dir",
            str(template_dir),
            "--output-dir",
            str(output_dir),
            "--rotation-lib",
            str(ROTATIONS_FILE),
        ]

        if dry_run:
            print("  [DRY RUN] Would execute:")
            print(f"  {' '.join(str(c) for c in cmd)}\n")
            continue

        run_cmd(cmd, f"Stage 2 (DWI phantom QC: {dwi_dir.name})")
        print()

    print("  Stage 2 complete.\n")


# ---------------------------------------------------------------------------
# Stage 3: Phantom QC on native contrasts
# ---------------------------------------------------------------------------


def run_stage3(
    input_dir: Path,
    output_dir: Path,
    template_dir: Path,
    scan_info: dict,
    dry_run: bool,
):
    """
    Convert T1, IR, and TE DICOMs to NIfTI into a staging folder, then
    run the phantom processor on the T1.  The phantom processor picks up
    all NIfTIs in the staging folder automatically.

    The staging folder is placed at:
        <output_dir>/<session_name>/native_contrasts_staging/

    After processing completes the staging folder is removed (unless
    dry_run is True).
    """
    print_header("STAGE 3 — Phantom QC on Native Contrasts")

    t1_dirs = scan_info["t1_dirs"]
    ir_dirs = scan_info["ir_dirs"]
    te_dirs = scan_info["te_dirs"]

    if not t1_dirs:
        print("  No T1 directory found — cannot run Stage 3.\n")
        return

    session_name = derive_session_name(input_dir)
    # Place the staging folder directly under output_dir so no intermediate
    # session subfolder is created (which would otherwise be left behind empty
    # after cleanup when session_name matches the output_dir's own name).
    staging_dir = output_dir / "native_contrasts_staging"

    print(f"  Session name:    {session_name}")
    print(f"  Staging folder:  {staging_dir}")
    print(f"  T1 dirs:         {[d.name for d in t1_dirs]}")
    print(f"  IR dirs:         {[d.name for d in ir_dirs]}")
    print(f"  TE dirs:         {[d.name for d in te_dirs]}")
    print()

    # --- Identify the primary T1 (first/only T1 directory) -----------------
    # For sessions with multiple T1 dirs, pick the one with the lowest series
    # number (matches how Stage 1 assigns T1s to DWI series).
    def _series_num(d: Path) -> int:
        m = re.match(r"^(\d+)-", d.name)
        return int(m.group(1)) if m else 0

    primary_t1_dir = sorted(t1_dirs, key=_series_num)[0]

    # --- Gather all contrast dirs to convert --------------------------------
    # Always include T1.  Include IR/TE if present.
    contrast_dirs_to_convert = [primary_t1_dir] + ir_dirs + te_dirs

    # In dry-run mode just describe what would happen
    if dry_run:
        print("  [DRY RUN] Would convert DICOMs to NIfTI and place in staging folder:")
        for d in contrast_dirs_to_convert:
            print(f"    {d.name}  →  {staging_dir}/")
        t1_nii_path = staging_dir / f"{primary_t1_dir.name}.nii.gz"
        cmd = [
            sys.executable,
            str(PHANTOM_SCRIPT),
            "single",
            str(t1_nii_path),
            "--template-dir",
            str(template_dir),
            "--output-dir",
            str(output_dir),
            "--rotation-lib",
            str(ROTATIONS_FILE),
        ]
        print("\n  [DRY RUN] Would execute:")
        print(f"  {' '.join(str(c) for c in cmd)}\n")
        return

    # --- Convert DICOMs to NIfTI ------------------------------------------
    staging_dir.mkdir(parents=True, exist_ok=True)
    t1_nii_path = None

    for dicom_dir in contrast_dirs_to_convert:
        print(f"  Converting DICOM: {dicom_dir.name}")
        try:
            produced = convert_dicom_dir(dicom_dir, staging_dir)
            print(f"    Produced: {[Path(p).name for p in produced]}")
            # The T1 NIfTI is from primary_t1_dir
            if dicom_dir == primary_t1_dir:
                # dcm2niix may produce multiple files; pick the first .nii.gz
                t1_nii_path = Path(produced[0])
        except Exception as e:
            print(f"  WARNING: DICOM conversion failed for {dicom_dir.name}: {e}")
            if dicom_dir == primary_t1_dir:
                print("  Cannot proceed with Stage 3 without a T1 image.")
                shutil.rmtree(staging_dir, ignore_errors=True)
                return

    if t1_nii_path is None or not t1_nii_path.exists():
        print("  ERROR: T1 NIfTI was not produced — aborting Stage 3.")
        shutil.rmtree(staging_dir, ignore_errors=True)
        return

    print(f"\n  All NIfTIs in staging folder:")
    for nii in sorted(staging_dir.glob("*.nii.gz")):
        print(f"    {nii.name}")

    # --- Run phantom processor --------------------------------------------
    # output_dir is passed as output_base_dir; session_name is derived
    # inside process_session() from input_path.parent.name, which will be
    # 'native_contrasts_staging'.  We therefore pass output_dir directly —
    # results land in <output_dir>/native_contrasts_staging/.
    # To get the cleaner <session_name> folder instead, we pass a one-level-
    # deeper output base so the phantom script writes to the right place.
    phantom_output_base = output_dir

    print(f"\n  Running phantom processor:")
    print(f"    Input image: {t1_nii_path}")
    print(f"    Output base: {phantom_output_base}")

    cmd = [
        sys.executable,
        str(PHANTOM_SCRIPT),
        "single",
        str(t1_nii_path),
        "--template-dir",
        str(template_dir),
        "--output-dir",
        str(phantom_output_base),
        "--rotation-lib",
        str(ROTATIONS_FILE),
    ]

    run_cmd(cmd, "Stage 3 (native phantom QC)")

    # --- Clean up staging NIfTIs only --------------------------------------
    # The phantom processor writes its outputs (metrics/, vial_segmentations/,
    # images_template_space/, TemplatePhantom_ScannerSpace.nii.gz) into
    # <output_dir>/native_contrasts_staging/ because it derives session_name
    # from the staging folder name.  We only delete the temporary NIfTI files
    # that were placed there for conversion — not the processed outputs.
    print(f"\n  Removing temporary NIfTIs from staging folder: {staging_dir}")
    for nii in staging_dir.glob("*.nii.gz"):
        nii.unlink(missing_ok=True)
    for jsn in staging_dir.glob("*.json"):
        jsn.unlink(missing_ok=True)
    # Remove the staging folder itself only if it is now empty
    try:
        staging_dir.rmdir()
        print(f"  Removed empty staging directory: {staging_dir.name}")
    except OSError:
        # Not empty — contains phantom outputs; leave it in place
        print(
            f"  Staging directory retained (contains phantom outputs): {staging_dir.name}"
        )

    print("\n  Stage 3 complete.\n")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_inputs(args):
    """Validate paths and phantom name before any processing."""
    errors = []

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        errors.append(f"--input-dir does not exist or is not a directory: {input_dir}")

    phantom_dir = TEMPLATE_DATA_ROOT / args.phantom
    if not phantom_dir.is_dir():
        errors.append(
            f"Phantom template directory not found: {phantom_dir}\n"
            f"  Expected: TemplateData/{args.phantom}/"
        )
    else:
        template_img = phantom_dir / "ImageTemplate.nii.gz"
        vials_dir = phantom_dir / "VialsLabelled"
        if not template_img.exists():
            errors.append(f"ImageTemplate.nii.gz not found in: {phantom_dir}")
        if not vials_dir.is_dir() or not list(vials_dir.glob("*.nii.gz")):
            errors.append(
                f"VialsLabelled/ with .nii.gz masks not found in: {phantom_dir}"
            )

    if not ROTATIONS_FILE.exists():
        errors.append(f"rotations.txt not found: {ROTATIONS_FILE}")

    if not DWI_SCRIPT.exists():
        errors.append(f"DWI processing script not found: {DWI_SCRIPT}")

    if not PHANTOM_SCRIPT.exists():
        errors.append(f"Phantom processing script not found: {PHANTOM_SCRIPT}")

    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end GSP phantom QC + DWI processing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing acquisition subdirectories (DICOM folders).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Top-level output directory.  All results are written here.",
    )
    parser.add_argument(
        "--phantom",
        required=True,
        help="Phantom name, e.g. SPIRIT.  Used to locate TemplateData/<phantom>/.",
    )

    # DWI pipeline flags (passed through to pydra_dwi_processing.py)
    parser.add_argument(
        "--denoise-degibbs",
        action="store_true",
        default=False,
        help="Apply dwidenoise + mrdegibbs before preprocessing.",
    )
    parser.add_argument(
        "--gradcheck",
        action="store_true",
        default=False,
        help="Run dwigradcheck to verify gradient orientations.",
    )
    parser.add_argument(
        "--nocleanup",
        action="store_true",
        default=False,
        help="Keep DWI tmp/ intermediate directories.",
    )
    parser.add_argument(
        "--readout-time",
        type=float,
        default=None,
        help="Override TotalReadoutTime (seconds) for dwifslpreproc.",
    )
    parser.add_argument(
        "--eddy-options",
        type=str,
        default=None,
        help="Override FSL eddy options string.",
    )

    # Orchestrator flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Plan and print commands; do not execute any processing.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    phantom = args.phantom
    template_dir = TEMPLATE_DATA_ROOT / phantom

    # Validate
    validate_inputs(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    # DWI pipeline config
    dwi_cfg = {
        "denoise_degibbs": args.denoise_degibbs,
        "gradcheck": args.gradcheck,
        "nocleanup": args.nocleanup,
        "readout_time": args.readout_time,
        "eddy_options": args.eddy_options,
    }

    # -----------------------------------------------------------------------
    # Discover what's in the input directory
    # -----------------------------------------------------------------------
    print_header("Input Directory Scan")
    scan_info = scan_input_dir(input_dir)

    print(f"  Input directory:     {input_dir}")
    print(f"  Phantom:             {phantom}")
    print(f"  Template dir:        {template_dir}")
    print(f"  Output dir:          {output_dir}")
    print()
    print(f"  T1 directories:      {len(scan_info['t1_dirs'])}")
    for d in scan_info["t1_dirs"]:
        print(f"    {d.name}")
    print(f"  IR directories:      {len(scan_info['ir_dirs'])}")
    for d in scan_info["ir_dirs"]:
        print(f"    {d.name}")
    print(f"  TE directories:      {len(scan_info['te_dirs'])}")
    for d in scan_info["te_dirs"]:
        print(f"    {d.name}")
    print(f"  DWI candidate dirs:  {len(scan_info['dwi_dirs'])}")
    for d in scan_info["dwi_dirs"]:
        print(f"    {d.name}")
    print(f"  Other (ignored):     {len(scan_info['other_dirs'])}")
    print()

    run_stage1_flag = scan_info["has_dwi"]
    run_stage3_flag = bool(scan_info["t1_dirs"]) and (
        scan_info["has_native_contrasts"] or not run_stage1_flag
    )

    print(
        f"  Stage 1 (DWI):                    {'YES' if run_stage1_flag else 'NO (no DWI found)'}"
    )
    print(
        f"  Stage 2 (phantom QC, DWI space):  {'YES (runs per DWI series)' if run_stage1_flag else 'NO'}"
    )
    print(
        f"  Stage 3 (phantom QC, native T1):  {'YES' if run_stage3_flag else 'NO (no T1/IR/TE found)'}"
    )
    print()

    if args.dry_run:
        print("  NOTE: --dry-run is active.  No processing will be performed.\n")

    # -----------------------------------------------------------------------
    # Execution
    #
    # Stage 1 (DWI) and Stage 3 (native contrasts) are independent and run
    # in parallel via a ThreadPoolExecutor.  Stage 2 (phantom QC in DWI
    # space) depends on Stage 1's outputs and runs after it completes.
    #
    # Execution graph:
    #   Stage 1 ──┐
    #              ├──→ Stage 2
    #   Stage 3 ──┘  (concurrent with Stage 1)
    # -----------------------------------------------------------------------

    # Thread-safe print lock so interleaved output from Stage 1 and Stage 3
    # is not garbled.
    _print_lock = threading.Lock()

    def _locked_print_header(title):
        with _print_lock:
            print_header(title)

    dwi_output_dirs = []
    stage1_error = None
    stage3_error = None

    def _run_stage1():
        nonlocal dwi_output_dirs, stage1_error
        try:
            if run_stage1_flag:
                dwi_output_dirs[:] = run_stage1(
                    input_dir, output_dir, dwi_cfg, args.dry_run
                )
            else:
                _locked_print_header("STAGE 1 — DWI Processing")
                with _print_lock:
                    print("  Skipped: no DWI acquisitions found.\n")
        except Exception as exc:
            stage1_error = exc

    def _run_stage3():
        nonlocal stage3_error
        try:
            if run_stage3_flag:
                run_stage3(input_dir, output_dir, template_dir, scan_info, args.dry_run)
            else:
                _locked_print_header("STAGE 3 — Phantom QC on Native Contrasts")
                with _print_lock:
                    if not scan_info["t1_dirs"]:
                        print("  Skipped: no T1 directory found.\n")
                    else:
                        print(
                            "  Skipped: no IR or TE series found, "
                            "and DWI pipeline was run.\n"
                        )
        except Exception as exc:
            stage3_error = exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_run_stage1)
        f3 = executor.submit(_run_stage3)
        # Wait for both to finish; exceptions are re-raised below
        concurrent.futures.wait([f1, f3])

    # Re-raise any errors after both threads have finished
    if stage1_error:
        raise RuntimeError(f"Stage 1 failed: {stage1_error}") from stage1_error
    if stage3_error:
        raise RuntimeError(f"Stage 3 failed: {stage3_error}") from stage3_error

    # Stage 2 runs after Stage 1 (sequential)
    if run_stage1_flag:
        run_stage2(dwi_output_dirs, output_dir, template_dir, args.dry_run)
    else:
        print_header("STAGE 2 — Phantom QC in DWI Space")
        print("  Skipped: Stage 1 did not run.\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print_header("Pipeline Complete")
    print(f"  All outputs written to: {output_dir}\n")

    if not args.dry_run:
        print("  Output structure:")
        for item in sorted(output_dir.iterdir()):
            if item.is_dir():
                print(f"    {item.name}/")
                for sub in sorted(item.iterdir()):
                    if sub.is_dir():
                        print(f"      {sub.name}/")
                    else:
                        print(f"      {sub.name}")
        print()


if __name__ == "__main__":
    main()
