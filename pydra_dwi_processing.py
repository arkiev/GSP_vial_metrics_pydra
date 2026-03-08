#!/usr/bin/env python3
"""
DWI processing and tensor metrics pipeline (Pydra 0.23+)

Usage:
    python pydra_dwi_processing.py --config pydra_dwi_processing.yaml
    python pydra_dwi_processing.py --scans-dir /path/to/scans --output-dir /path/to/out [options]

Output structure:
    <output_dir>/
        <DWI_series_name>/
            ADC.nii.gz
            FA.nii.gz
            DWI_preproc_biascorr.mif.gz   (or DWI_denoise_gibbs_preproc_biascorr.mif.gz)
            T1_n4_in_DWI_space.nii.gz
            tmp/                           (all intermediate files)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pydra
import yaml


# =============================================================================
# Utility helpers
# =============================================================================


def run_cmd(cmd: list, cwd: str = None):
    """Run a shell command, raising on failure."""
    print(f"  >> {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def sanitise_name(name: str) -> str:
    """Convert a folder name to a valid Python identifier for Pydra task names."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def strip_series_number(name: str) -> str:
    """Strip leading series number, e.g. '87-ep2d_diff_...' -> 'ep2d_diff_...'"""
    return re.sub(r"^\d+-", "", name)


def get_series_number(name: str) -> int:
    """Extract leading series number from folder name, e.g. '87-ep2d...' -> 87."""
    m = re.match(r"^(\d+)-", name)
    return int(m.group(1)) if m else 0


def common_prefix_len(a: str, b: str) -> int:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def get_nvols(path: str) -> int:
    """Return number of volumes using mrinfo."""
    result = subprocess.run(
        ["mrinfo", path, "-ndim"], capture_output=True, text=True, check=True
    )
    ndim = int(result.stdout.strip())
    if ndim == 3:
        return 1
    result2 = subprocess.run(
        ["mrinfo", path, "-size"], capture_output=True, text=True, check=True
    )
    return int(result2.stdout.strip().split()[-1])


def read_bvals(bval_path: str) -> list:
    """Read bval file and return list of float values. Returns [] if missing/empty."""
    try:
        with open(bval_path) as f:
            content = f.read().strip()
        if not content:
            return []
        return [float(v) for v in content.split()]
    except Exception:
        return []


def has_nonzero_bvals(bval_path: str) -> bool:
    """Return True if bval file exists, is non-empty, and contains non-zero values."""
    vals = read_bvals(bval_path)
    return bool(vals) and any(v > 10 for v in vals)  # threshold >10 to ignore rounding


def all_zero_bvals(bval_path: str) -> bool:
    """Return True if bval file exists and all values are effectively zero."""
    vals = read_bvals(bval_path)
    return bool(vals) and all(v <= 10 for v in vals)


def get_readout_time(json_path: str, fallback: float = 0.0342002) -> float:
    """Extract TotalReadoutTime from dcm2niix JSON sidecar."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        val = data.get("TotalReadoutTime")
        if val is not None:
            return float(val)
    except Exception:
        pass
    print(f"  Warning: TotalReadoutTime not found in {json_path} -- using {fallback}")
    return fallback


def get_pe_from_json(json_path: str) -> tuple:
    """
    (#9) Fallback: read PhaseEncodingDirection from JSON sidecar and map to
    MRtrix3-compatible direction string.
    Returns (pe_dir, rpe_dir) or raises ValueError if not found/unrecognised.
    """
    mapping = {
        "j-": ("AP", "PA"),
        "j": ("PA", "AP"),
        "i": ("LR", "RL"),
        "i-": ("RL", "LR"),
        "k": ("SI", "IS"),
        "k-": ("IS", "SI"),
    }
    try:
        with open(json_path) as f:
            data = json.load(f)
        ped = data.get("PhaseEncodingDirection", "").strip()
        if ped in mapping:
            return mapping[ped]
    except Exception:
        pass
    raise ValueError(f"PhaseEncodingDirection not found or unrecognised in {json_path}")


def detect_pe_direction(folder_name: str) -> tuple:
    """
    Return (pe_dir, rpe_dir) from a folder name.
    Raises ValueError if not found — caller should fall back to get_pe_from_json (#9).
    """
    pairs = [
        (r"_AP(_|$)", "AP", "PA"),
        (r"_A_P(_|$)", "AP", "PA"),
        (r"_PA(_|$)", "PA", "AP"),
        (r"_P_A(_|$)", "PA", "AP"),
        (r"_LR(_|$)", "LR", "RL"),
        (r"_L_R(_|$)", "LR", "RL"),
        (r"_RL(_|$)", "RL", "LR"),
        (r"_R_L(_|$)", "RL", "LR"),
        (r"_SI(_|$)", "SI", "IS"),
        (r"_S_I(_|$)", "SI", "IS"),
        (r"_IS(_|$)", "IS", "SI"),
        (r"_I_S(_|$)", "IS", "SI"),
    ]
    for pattern, pe, rpe in pairs:
        if re.search(pattern, folder_name, re.IGNORECASE):
            return pe, rpe
    raise ValueError(
        f"Could not determine phase encoding direction from folder name: {folder_name}"
    )


# =============================================================================
# Directory scanning constants
# =============================================================================

FWD_DIRS = ["AP", "LR", "SI", "A_P", "L_R", "S_I"]
RPE_DIRS = ["PA", "RL", "IS", "P_A", "R_L", "I_S"]
ALL_PE_DIRS = FWD_DIRS + RPE_DIRS


def has_terminal_pe(name: str, directions: list) -> bool:
    """
    Returns True if a PE direction tag appears as the final token in the folder
    name (nothing follows, or only a single short alphanumeric token with no
    underscores). This identifies dedicated b0 PE images vs full DWI series.
    """
    for d_ in directions:
        for pat in [rf"_{d_}$", rf"_{d_}_[A-Za-z0-9]{{1,10}}$"]:
            if re.search(pat, name, re.IGNORECASE):
                return True
        # Underscore-separated variant (e.g. A_P, P_A)
        if len(d_) == 3 and d_[1] == "_":
            for pat in [
                rf"_{d_[0]}_{d_[2]}$",
                rf"_{d_[0]}_{d_[2]}_[A-Za-z0-9]{{1,10}}$",
            ]:
                if re.search(pat, name, re.IGNORECASE):
                    return True
    return False


def get_acq_stem_and_suffix(folder_name: str) -> tuple:
    """
    Split acquisition name into (prefix, suffix) around the PE direction tag.
    Both must match exactly for two series to be considered valid AP/PA partners
    (#14 — prevents pairing of MONO/BIPOLAR variants or different parameters).

    e.g. '12-ep2d_diff__FREE30DIR_b1000_x7b0_Ghost_R_AP_ESpt54_BW2264_BIPOLAR'
      -> prefix: 'ep2d_diff__FREE30DIR_b1000_x7b0_Ghost_R'
         suffix: 'ESpt54_BW2264_BIPOLAR'

    Returns (full_name_no_series, "", "") if no PE direction tag found.
    """
    name = strip_series_number(folder_name)
    # Try longer variants first (A_P before AP) to avoid partial matches
    all_dirs_ordered = [
        "A_P",
        "P_A",
        "L_R",
        "R_L",
        "S_I",
        "I_S",
        "AP",
        "PA",
        "LR",
        "RL",
        "SI",
        "IS",
    ]
    for d in all_dirs_ordered:
        m = re.search(rf"_({re.escape(d)})(_|$)", name, re.IGNORECASE)
        if m:
            prefix = name[: m.start()]
            # suffix is everything after the direction tag (and the following _)
            suffix_start = m.end()
            suffix = name[suffix_start:] if suffix_start < len(name) else ""
            return prefix, suffix
    return name, ""


# =============================================================================
# DICOM conversion helper (used in planning stage)
# =============================================================================


def convert_dicom_to_nii(dicom_dir: str, out_dir: str) -> dict:
    """
    Run dcm2niix on a DICOM directory, returning a dict with paths to:
      nii, json, bvec, bval
    Returns empty strings for missing files.
    """
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["dcm2niix", "-o", out_dir, "-f", "%p", "-z", "y", dicom_dir],
        check=True,
        capture_output=True,
    )
    niis = sorted(Path(out_dir).glob("*.nii.gz"))
    if not niis:
        raise FileNotFoundError(f"No NIfTI produced from DICOM: {dicom_dir}")
    nii = str(niis[0])
    base = nii.replace(".nii.gz", "")
    return {
        "nii": nii,
        "json": base + ".json" if Path(base + ".json").exists() else "",
        "bvec": base + ".bvec" if Path(base + ".bvec").exists() else "",
        "bval": base + ".bval" if Path(base + ".bval").exists() else "",
    }


# =============================================================================
# Directory scanning and candidate classification
# =============================================================================


def scan_directory(scans_dir: str) -> dict:
    """
    First pass: classify subdirectories into t1_dirs, candidate_dwi (all series
    containing _diff_ or _DWI_), and ignored. PE classification is PROVISIONAL
    at this stage — final assignment happens after DICOM conversion in
    classify_candidates().
    """
    scans_path = Path(scans_dir)
    t1_dirs = []
    candidate_dwi = []  # all _diff_ / _DWI_ series, including PE images
    ignored = []

    for d in sorted(scans_path.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        # T1
        if re.search(r"t1", name, re.IGNORECASE):
            t1_dirs.append(str(d))
            continue

        # Skip scanner-derived maps
        if re.search(r"_(ADC|FA)$", name, re.IGNORECASE):
            ignored.append(str(d))
            continue

        # Only consider DWI-like acquisitions
        if not re.search(r"(_diff_|_DWI_)", name, re.IGNORECASE):
            ignored.append(str(d))
            continue

        candidate_dwi.append(str(d))

    if not t1_dirs:
        raise ValueError(f"Could not identify any T1 directory in {scans_dir}")
    if not candidate_dwi:
        raise ValueError(f"Could not identify any DWI directories in {scans_dir}")

    return {
        "t1_dirs": t1_dirs,
        "candidate_dwi": candidate_dwi,
        "ignored": ignored,
    }


def convert_all_candidates(candidate_dwi: list, output_dir: str) -> dict:
    """
    (#18 ordering step 2) Convert all candidate DWI DICOMs upfront.
    Returns a dict: {dicom_dir: {nii, json, bvec, bval}}
    """
    conversions = {}
    for dicom_dir in candidate_dwi:
        name = Path(dicom_dir).name
        conv_dir = str(Path(output_dir) / name / "tmp" / "dwi_nii")
        print(f"  Converting: {name}")
        try:
            result = convert_dicom_to_nii(dicom_dir, conv_dir)
            conversions[dicom_dir] = result
        except Exception as e:
            print(f"  WARNING: DICOM conversion failed for {name}: {e}")
            conversions[dicom_dir] = None
    return conversions


def classify_candidates(candidate_dwi: list, conversions: dict) -> dict:
    """
    (#18) Final classification of candidate DWI series into:
      - dwi_dirs:     full DWI series eligible for tensor processing
      - fwd_pe_dirs:  forward PE correction images (b0-only or low-volume)
      - rpe_dirs:     reverse PE correction images (b0-only or low-volume)
      - skipped:      list of (path, reason) tuples

    Classification rules (applied in order after DICOM conversion):
      1. No bvec/bval files → skip (#16)
      2. All b-values zero → candidate PE correction image, classify by PE tag
      3. Has non-zero b-values AND terminal PE tag → provisional PE image;
         will be reclassified to dwi_dirs if volume count matches a partner
      4. Has non-zero b-values, no terminal PE tag → dwi_dirs
    """
    dwi_dirs = []
    fwd_pe_dirs = []
    rpe_dirs = []
    skipped = []
    # pending: series with terminal PE tag and non-zero bvals — reclassified after pairing
    pending_fwd = []
    pending_rpe = []

    for dicom_dir in candidate_dwi:
        name = Path(dicom_dir).name
        conv = conversions.get(dicom_dir)

        if conv is None:
            skipped.append((dicom_dir, "DICOM conversion failed"))
            continue

        bvec = conv["bvec"]
        bval = conv["bval"]

        # #16 — check bvec/bval exist and are non-empty
        if not bvec or not bval or not Path(bvec).exists() or not Path(bval).exists():
            skipped.append(
                (dicom_dir, "no bvec/bval files — not a raw DWI acquisition")
            )
            continue
        if not read_bvals(bval):
            skipped.append((dicom_dir, "empty bval file — not a raw DWI acquisition"))
            continue

        is_b0_only = all_zero_bvals(bval)
        terminal_fwd = has_terminal_pe(name, FWD_DIRS)
        terminal_rpe = has_terminal_pe(name, RPE_DIRS)

        if is_b0_only:
            # All b-values zero: PE correction image candidate
            if terminal_fwd:
                fwd_pe_dirs.append(dicom_dir)
            elif terminal_rpe:
                rpe_dirs.append(dicom_dir)
            else:
                # No direction tag — note in plan but cannot use for correction
                skipped.append(
                    (
                        dicom_dir,
                        "all b-values zero and no phase encoding direction in folder name — "
                        "cannot use for correction",
                    )
                )
        else:
            # Has non-zero b-values
            if terminal_fwd:
                pending_fwd.append(dicom_dir)
            elif terminal_rpe:
                pending_rpe.append(dicom_dir)
            else:
                dwi_dirs.append(dicom_dir)

    return {
        "dwi_dirs": dwi_dirs,
        "fwd_pe_dirs": fwd_pe_dirs,
        "rpe_dirs": rpe_dirs,
        "pending_fwd": pending_fwd,  # terminal PE tag + non-zero bvals, needs pairing
        "pending_rpe": pending_rpe,
        "skipped": skipped,
    }


# =============================================================================
# AP/PA pairing
# =============================================================================


def match_ap_pa_pairs(
    dwi_dirs: list, pending_fwd: list, pending_rpe: list, conversions: dict
) -> tuple:
    """
    (#14, #18) Detect AP/PA (rpe_all/rpe_split) pairs across dwi_dirs,
    pending_fwd, and pending_rpe.

    Pairing requires:
      - Same acquisition prefix (everything before the PE direction tag)
      - Same acquisition suffix (everything after the PE direction tag) — #14
      - Opposing PE directions

    Pending series with non-zero bvals that find a partner are moved to dwi_dirs.
    Pending series that find no partner are moved to fwd_pe_dirs / rpe_dirs.

    Returns:
      (dwi_dirs, fwd_pe_dirs, rpe_dirs, rpe_all_map, tie_warnings)
      rpe_all_map: {fwd_path: rpe_path}
      tie_warnings: list of (pe_image_path, [matched_dwi_paths]) for #20
    """
    fwd_set = {"AP", "LR", "SI"}
    rpe_set = {"PA", "RL", "IS"}

    # Build stem map over all candidates (dwi + pending)
    all_candidates = dwi_dirs + pending_fwd + pending_rpe
    stem_map = {}  # (prefix, suffix) -> list of (path, pe_dir)
    for d in all_candidates:
        name = Path(d).name
        try:
            pe, _ = detect_pe_direction(name)
        except ValueError:
            pe = None
        prefix, suffix = get_acq_stem_and_suffix(name)
        key = (prefix, suffix)
        stem_map.setdefault(key, []).append((d, pe))

    rpe_all_map = {}
    tie_warnings = []
    paired_paths = set()

    for (prefix, suffix), entries in stem_map.items():
        fwds = [(p, pe) for p, pe in entries if pe in fwd_set]
        rpes = [(p, pe) for p, pe in entries if pe in rpe_set]

        if not (fwds and rpes):
            continue

        for fwd_path, _ in fwds:
            fwd_num = get_series_number(Path(fwd_path).name)
            # Find closest RPE by series number
            best_rpe_path = min(
                rpes, key=lambda x: abs(get_series_number(Path(x[0]).name) - fwd_num)
            )[0]
            rpe_all_map[fwd_path] = best_rpe_path
            paired_paths.add(fwd_path)
            paired_paths.add(best_rpe_path)

            # #20 — check for tie (multiple equal-prefix matches)
            if len(fwds) > 1 or len(rpes) > 1:
                tie_warnings.append((fwd_path, [p for p, _ in rpes]))

    # Move paired pending series into dwi_dirs; unpaired pending into PE dirs
    final_dwi = list(dwi_dirs)
    final_fwd = list(conversions.get("fwd_pe_dirs", []))
    final_rpe = list(conversions.get("rpe_dirs", []))

    for p in pending_fwd:
        if p in paired_paths:
            final_dwi.append(p)
        else:
            final_fwd.append(p)

    for p in pending_rpe:
        if p in paired_paths:
            # PA partner of a paired series — recorded in rpe_all_map, not in dwi_dirs
            pass
        else:
            final_rpe.append(p)

    return sorted(final_dwi), final_fwd, final_rpe, rpe_all_map, tie_warnings


# =============================================================================
# T1 and PE assignment helpers
# =============================================================================


def assign_t1(dwi_name: str, t1_dirs: list) -> str:
    """
    Assign the nearest preceding T1 by series number.
    Falls back to the first T1 if none precedes the DWI.
    """
    dwi_num = get_series_number(dwi_name)
    best = None
    best_num = -1
    for t1 in t1_dirs:
        t1_num = get_series_number(Path(t1).name)
        if t1_num < dwi_num and t1_num > best_num:
            best_num = t1_num
            best = t1
    return best if best is not None else t1_dirs[0]


def find_best_pe_match(dwi_name: str, pe_dirs: list) -> tuple:
    """
    (#19, #20) Match a PE directory to a DWI by longest common prefix.
    When multiple PE dirs tie, break by nearest series number.

    Returns (best_dir, tie_warning) where tie_warning is True if multiple
    dirs shared the same best prefix length.
    """
    dwi_stem = strip_series_number(dwi_name)
    dwi_num = get_series_number(dwi_name)
    best_len = 0
    best_dirs = []

    for d in pe_dirs:
        pe_stem = strip_series_number(Path(d).name)
        length = common_prefix_len(dwi_stem, pe_stem)
        if length > best_len:
            best_len = length
            best_dirs = [d]
        elif length == best_len and length > 0:
            best_dirs.append(d)

    if not best_dirs or best_len <= 10:
        # Weak match — fall back to nearest series number
        best_dirs = pe_dirs
        tie = len(pe_dirs) > 1
    else:
        tie = len(best_dirs) > 1

    # Break tie by nearest series number
    best = min(best_dirs, key=lambda d: abs(get_series_number(Path(d).name) - dwi_num))
    return best, tie


# =============================================================================
# Plan construction
# =============================================================================


def resolve_pe_direction(dwi_name: str, dwi_json: str) -> tuple:
    """
    (#9) Try to determine PE direction from folder name first.
    Fall back to JSON sidecar if not found. Returns (pe_dir, rpe_dir, source)
    where source is 'folder_name' or 'json_sidecar'.
    Raises ValueError if neither source yields a direction.
    """
    try:
        pe_dir, rpe_dir = detect_pe_direction(dwi_name)
        return pe_dir, rpe_dir, "folder_name"
    except ValueError:
        pass

    if dwi_json and Path(dwi_json).exists():
        try:
            pe_dir, rpe_dir = get_pe_from_json(dwi_json)
            return pe_dir, rpe_dir, "json_sidecar"
        except ValueError:
            pass

    raise ValueError(
        f"Could not determine phase encoding direction for {dwi_name} "
        f"from folder name or JSON sidecar."
    )


def plan_workflow(
    dwi_dir: str,
    t1_dirs: list,
    fwd_pe_dirs: list,
    rpe_dirs: list,
    rpe_all_map: dict,
    tie_warnings: list,
    conversions: dict,
    cfg: dict,
) -> dict:
    """
    Build a complete plan dict for a single DWI series.
    All DICOM conversion has already been done — we use results from conversions.
    """
    dwi_name = Path(dwi_dir).name
    conv = conversions[dwi_dir]
    dwi_nii = conv["nii"]
    dwi_json = conv["json"]
    dwi_bvec = conv["bvec"]
    dwi_bval = conv["bval"]
    dwi_nvols = get_nvols(dwi_nii)

    out_dir = str(Path(cfg["output_dir"]) / dwi_name)
    tmp_dir = str(Path(out_dir) / "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # T1 assignment
    t1_dir = assign_t1(dwi_name, t1_dirs)

    # PE direction (#9)
    warnings = []
    try:
        pe_dir, rpe_dir, pe_source = resolve_pe_direction(dwi_name, dwi_json)
    except ValueError as e:
        raise ValueError(str(e))

    if pe_source == "json_sidecar":
        warnings.append(
            f"Phase encoding direction ({pe_dir}) inferred from JSON sidecar "
            f"— not found in folder name. Please verify."
        )

    # RPE assignment
    rpe_all_partner = rpe_all_map.get(dwi_dir)

    rpe_nii = None
    rpe_nvols = 0
    fwd_pe_nii = None
    fwd_pe_dir_used = None
    rpe_dir_path = None

    if rpe_all_partner:
        # rpe_all: full-volume PA partner
        rpe_dir_path = rpe_all_partner
        rpe_conv = conversions.get(rpe_all_partner)
        if rpe_conv:
            rpe_nii = rpe_conv["nii"]
            rpe_nvols = get_nvols(rpe_nii)

        # Check for rpe_all vs rpe_split warning
        rpe_all_warning = None
        if rpe_nvols == dwi_nvols:
            rpe_all_warning = (
                f"{dwi_name} and {Path(rpe_all_partner).name} have equal volumes "
                f"({dwi_nvols}). Assuming rpe_all (full repeats). If directions are "
                f"split across PE directions, set mode: rpe_split in config YAML."
            )
            warnings.append(rpe_all_warning)

        fwd_pe_dir_used = None  # b0 PE images not needed

    else:
        # Use b0 PE correction images
        rpe_match = None
        rpe_tie = False
        if rpe_dirs:
            rpe_match, rpe_tie = find_best_pe_match(dwi_name, rpe_dirs)

        fwd_match = None
        fwd_tie = False
        if fwd_pe_dirs:
            fwd_match, fwd_tie = find_best_pe_match(dwi_name, fwd_pe_dirs)

        rpe_dir_path = rpe_match
        fwd_pe_dir_used = fwd_match

        # #20 — tie warnings
        if rpe_tie and rpe_match:
            warnings.append(
                f"{Path(rpe_match).name} matched equally to multiple DWI series. "
                f"Assigned by nearest series number. If incorrect, consider adding "
                f"a distinguishing suffix (e.g. _repeat) to the second acquisition "
                f"block and its correction pair."
            )
        if fwd_tie and fwd_match:
            warnings.append(
                f"{Path(fwd_match).name} matched equally to multiple DWI series. "
                f"Assigned by nearest series number."
            )

        # Convert RPE if available
        if rpe_dir_path:
            rpe_conv_dir = str(Path(tmp_dir) / "rpe_nii")
            rpe_conv = conversions.get(rpe_dir_path)
            if rpe_conv is None:
                # Convert now if not already done
                rpe_conv = convert_dicom_to_nii(rpe_dir_path, rpe_conv_dir)
                conversions[rpe_dir_path] = rpe_conv
            rpe_nii = rpe_conv["nii"]
            rpe_nvols = get_nvols(rpe_nii)

        # Convert FWD PE if available
        if fwd_pe_dir_used:
            fwd_conv_dir = str(Path(tmp_dir) / "fwd_pe_nii")
            fwd_conv = conversions.get(fwd_pe_dir_used)
            if fwd_conv is None:
                fwd_conv = convert_dicom_to_nii(fwd_pe_dir_used, fwd_conv_dir)
                conversions[fwd_pe_dir_used] = fwd_conv
            fwd_pe_nii = fwd_conv["nii"]

        # #17 — orphaned RPE handling
        if rpe_dir_path and not fwd_pe_dir_used:
            if rpe_nvols == 1:
                warnings.append(
                    f"{Path(rpe_dir_path).name} has no matching forward PE partner. "
                    f"Will use mean b0 from main DWI as forward PE image (rpe_pair)."
                )
            elif rpe_nvols > 1:
                warnings.append(
                    f"{Path(rpe_dir_path).name} has no matching forward PE partner "
                    f"and RPE has multiple volumes. Cannot use for correction. "
                    f"Falling back to rpe_none."
                )
                rpe_dir_path = None
                rpe_nii = None
                rpe_nvols = 0

    # Determine preproc mode
    if rpe_nii is None:
        preproc_mode = "rpe_none"
    elif rpe_nvols == dwi_nvols:
        preproc_mode = "rpe_all"
    elif rpe_nvols > 1 and fwd_pe_nii is not None:
        preproc_mode = "rpe_split"
    else:
        preproc_mode = "rpe_pair"

    # #21 — note unused PE images
    notes = []
    if rpe_all_partner:
        for d in fwd_pe_dirs + rpe_dirs:
            notes.append(f"{Path(d).name} — not used (DWI series has rpe_all partner)")

    # Resolve readout time
    readout_time_override = cfg.get("readout_time", None)
    if readout_time_override is not None:
        readout_time = float(readout_time_override)
        readout_time_source = "config override"
    elif dwi_json and Path(dwi_json).exists():
        readout_time = get_readout_time(dwi_json)
        readout_time_source = f"JSON ({Path(dwi_json).name})"
    else:
        readout_time = 0.0342002
        readout_time_source = "fallback (no JSON found)"

    do_denoise = cfg.get("denoise_degibbs", False)
    do_gradcheck = cfg.get("gradcheck", False)
    eddy_options = cfg.get("eddy_options", " --slm=linear")

    dwi_preproc_name = (
        "DWI_denoise_gibbs_preproc_biascorr.mif.gz"
        if do_denoise
        else "DWI_preproc_biascorr.mif.gz"
    )

    # Build paths for sidecar files
    rpe_conv = conversions.get(rpe_dir_path) if rpe_dir_path else None
    fwd_conv = conversions.get(fwd_pe_dir_used) if fwd_pe_dir_used else None

    return {
        "dwi_name": dwi_name,
        "dwi_dir": dwi_dir,
        "t1_dir": t1_dir,
        "fwd_pe_dir": fwd_pe_dir_used,
        "rpe_dir_path": rpe_dir_path,
        "pe_dir": pe_dir,
        "rpe_dir": rpe_dir,
        "pe_source": pe_source,
        "dwi_nii": dwi_nii,
        "dwi_json": dwi_json,
        "dwi_bvec": dwi_bvec,
        "dwi_bval": dwi_bval,
        "dwi_nvols": dwi_nvols,
        "rpe_nii": rpe_nii,
        "rpe_json": rpe_conv["json"] if rpe_conv else "",
        "rpe_bvec": rpe_conv["bvec"] if rpe_conv else "",
        "rpe_bval": rpe_conv["bval"] if rpe_conv else "",
        "rpe_nvols": rpe_nvols,
        "fwd_pe_nii": fwd_pe_nii,
        "fwd_pe_json": fwd_conv["json"] if fwd_conv else "",
        "fwd_pe_bvec": fwd_conv["bvec"] if fwd_conv else "",
        "fwd_pe_bval": fwd_conv["bval"] if fwd_conv else "",
        "preproc_mode": preproc_mode,
        "readout_time": readout_time,
        "readout_time_src": readout_time_source,
        "do_denoise": do_denoise,
        "do_gradcheck": do_gradcheck,
        "eddy_options": eddy_options,
        "dwi_preproc_name": dwi_preproc_name,
        "out_dir": out_dir,
        "tmp_dir": tmp_dir,
        "warnings": warnings,
        "notes": notes,
    }


def print_plan(plans: list, skipped: list):
    """Print workflow summary including all warnings and notes."""
    print("\n" + "=" * 70)
    print("WORKFLOW PLAN")
    print("=" * 70)

    if skipped:
        print("\nSKIPPED SERIES:")
        for path, reason in skipped:
            print(f"  SKIPPED: {Path(path).name}")
            print(f"           ({reason})")

    for p in plans:
        rpe_name = Path(p["rpe_dir_path"]).name if p["rpe_dir_path"] else "not found"
        fwd_name = Path(p["fwd_pe_dir"]).name if p["fwd_pe_dir"] else "not used"

        print(f"\nDWI series:       {p['dwi_name']}")
        print(f"  T1:             {Path(p['t1_dir']).name}")

        if p["preproc_mode"] == "rpe_all":
            print(f"  AP/PA pair:     {p['dwi_name']}  ({p['dwi_nvols']} vols)")
            print(f"                  {rpe_name}  ({p['rpe_nvols']} vols)")
            print(f"  Forward PE b0:  not used (rpe_all)")
            print(f"  Reverse PE b0:  not used (rpe_all)")
        else:
            print(f"  Forward PE:     {fwd_name}")
            print(f"  Reverse PE:     {rpe_name}")
            print(f"  DWI volumes:    {p['dwi_nvols']}")
            print(f"  RPE volumes:    {p['rpe_nvols']}")

        pe_str = p["pe_dir"]
        if p["pe_source"] == "json_sidecar":
            pe_str += (
                "  (WARNING: inferred from JSON sidecar — not found in folder name)"
            )
        print(f"  PE direction:   {pe_str}")
        print(f"  Preproc mode:   {p['preproc_mode']}")
        print(f"  Readout time:   {p['readout_time']} ({p['readout_time_src']})")
        print(f"  Denoise/Gibbs:  {p['do_denoise']}")
        print(f"  Gradcheck:      {p['do_gradcheck']}")

        if p["do_gradcheck"]:
            targets = [f"DWI ({p['pe_dir']})"]
            if p["preproc_mode"] == "rpe_all":
                targets.append(f"DWI ({p['rpe_dir']})")
            elif p["fwd_pe_nii"]:
                targets.append("Forward PE b0")
            if p["rpe_nii"] and p["preproc_mode"] != "rpe_all":
                targets.append("Reverse PE b0")
            print(f"    Gradcheck on: {', '.join(targets)}")

        if p["do_denoise"] and p["preproc_mode"] == "rpe_all":
            print(
                f"    Denoise on:   DWI ({p['pe_dir']}) and DWI ({p['rpe_dir']}) separately"
            )

        print(f"  Output:         {p['out_dir']}")

        for w in p["warnings"]:
            print(f"  WARNING: {w}")
        for n in p["notes"]:
            print(f"  NOTE:    {n}")

    print("\n" + "=" * 70)
    print("Proceed? (Ctrl+C to abort)")
    print("=" * 70 + "\n")


# =============================================================================
# Pydra tasks
# =============================================================================


@pydra.mark.task
@pydra.mark.annotate({"return": {"nii": str, "json": str, "bvec": str, "bval": str}})
def convert_dicoms(dicom_dir: str, out_dir: str):
    """Convert a DICOM directory to NIfTI. Returns NIfTI + sidecar paths."""
    os.makedirs(out_dir, exist_ok=True)
    run_cmd(["dcm2niix", "-o", out_dir, "-f", "%p", "-z", "y", dicom_dir])
    niis = sorted(Path(out_dir).glob("*.nii.gz"))
    if not niis:
        raise FileNotFoundError(f"No NIfTI files produced in {out_dir}")
    nii = str(niis[0])
    base = nii.replace(".nii.gz", "")
    return (
        nii,
        base + ".json" if Path(base + ".json").exists() else "",
        base + ".bvec" if Path(base + ".bvec").exists() else "",
        base + ".bval" if Path(base + ".bval").exists() else "",
    )


@pydra.mark.task
def convert_to_mif_initial(
    nii: str, json_file: str, bvec: str, bval: str, out_path: str
) -> str:
    """Step 1 of gradcheck: mrconvert NIfTI -> MIF with original gradients embedded."""
    run_cmd(
        ["mrconvert", nii, out_path, "-json_import", json_file, "-fslgrad", bvec, bval]
    )
    return out_path


@pydra.mark.task
@pydra.mark.annotate({"return": {"bvec": str, "bval": str}})
def run_gradcheck(mif_path: str, out_dir: str, prefix: str):
    """Run dwigradcheck on a MIF, export corrected bvec/bval."""
    out_bvec = str(Path(out_dir) / f"{prefix}_corrected.bvec")
    out_bval = str(Path(out_dir) / f"{prefix}_corrected.bval")
    run_cmd(["dwigradcheck", mif_path, "-export_grad_fsl", out_bvec, out_bval])
    return (out_bvec, out_bval)


@pydra.mark.task
def convert_to_mif_final(
    nii: str, json_file: str, bvec: str, bval: str, out_path: str
) -> str:
    """mrconvert NIfTI -> MIF with (optionally corrected) gradients."""
    run_cmd(
        ["mrconvert", nii, out_path, "-json_import", json_file, "-fslgrad", bvec, bval]
    )
    result = subprocess.run(
        ["mrinfo", out_path, "-petable"], capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"  Warning: PE table not found in MIF header: {out_path}")
    return out_path


@pydra.mark.task
def run_dwidenoise(in_mif: str, out_mif: str) -> str:
    run_cmd(["dwidenoise", in_mif, out_mif])
    return out_mif


@pydra.mark.task
def run_mrdegibbs(in_mif: str, out_mif: str) -> str:
    run_cmd(["mrdegibbs", in_mif, out_mif])
    return out_mif


@pydra.mark.task
def concatenate_ap_pa(ap_mif: str, pa_mif: str, out_mif: str) -> str:
    """Concatenate AP and PA DWI series (AP first) for rpe_all."""
    run_cmd(["dwicat", ap_mif, pa_mif, out_mif])
    return out_mif


@pydra.mark.task
def build_se_epi(
    dwi_mif: str,
    rpe_nii: str,
    rpe_json: str,
    rpe_bvec: str,
    rpe_bval: str,
    fwd_pe_nii: str,
    fwd_pe_json: str,
    fwd_pe_bvec: str,
    fwd_pe_bval: str,
    pe_dir: str,
    rpe_dir: str,
    preproc_mode: str,
    tmp_dir: str,
) -> str:
    """Build se_epi pair for rpe_pair / rpe_split. Returns bzero_pair path."""
    if preproc_mode not in ("rpe_pair", "rpe_split"):
        return ""

    rpe_mif = str(Path(tmp_dir) / f"DWI_ref_{rpe_dir}.mif.gz")
    run_cmd(
        [
            "mrconvert",
            rpe_nii,
            rpe_mif,
            "-json_import",
            rpe_json,
            "-fslgrad",
            rpe_bvec,
            rpe_bval,
        ]
    )

    if fwd_pe_nii:
        fwd_mif = str(Path(tmp_dir) / f"DWI_ref_{pe_dir}.mif.gz")
        run_cmd(
            [
                "mrconvert",
                fwd_pe_nii,
                fwd_mif,
                "-json_import",
                fwd_pe_json,
                "-fslgrad",
                fwd_pe_bvec,
                fwd_pe_bval,
            ]
        )
    else:
        # #17 — use mean b0 from main DWI as forward PE
        print("  No forward PE image — computing mean b0 from DWI...")
        mean_b0 = str(Path(tmp_dir) / f"mean_bzero_{pe_dir}.mif.gz")
        extract = subprocess.Popen(
            ["dwiextract", dwi_mif, "-", "-bzero"], stdout=subprocess.PIPE
        )
        subprocess.run(
            ["mrmath", "-", "mean", mean_b0, "-axis", "3"],
            stdin=extract.stdout,
            check=True,
        )
        extract.stdout.close()
        extract.wait()
        fwd_mif = mean_b0

    bzero_pair = str(Path(tmp_dir) / "bzero_pair.mif.gz")
    run_cmd(["dwicat", fwd_mif, rpe_mif, bzero_pair])
    return bzero_pair


@pydra.mark.task
def run_dwifslpreproc(
    dwi_mif: str,
    out_mif: str,
    pe_dir: str,
    preproc_mode: str,
    se_epi: str,
    readout_time: float,
    eddy_options: str,
) -> str:
    cmd = [
        "dwifslpreproc",
        dwi_mif,
        out_mif,
        "-pe_dir",
        pe_dir,
        f"-{preproc_mode}",
        "-eddy_options",
        eddy_options,
    ]
    if preproc_mode in ("rpe_pair", "rpe_split"):
        cmd += ["-se_epi", se_epi, "-readout_time", str(readout_time), "-align_seepi"]
    elif preproc_mode == "rpe_all":
        cmd += ["-readout_time", str(readout_time)]
    run_cmd(cmd)
    return out_mif


@pydra.mark.task
def run_dwi2mask(in_mif: str, out_mif: str) -> str:
    run_cmd(["dwi2mask", in_mif, out_mif])
    return out_mif


@pydra.mark.task
def run_dwibiascorrect(in_mif: str, out_mif: str, mask_mif: str, bias_mif: str) -> str:
    run_cmd(
        [
            "dwibiascorrect",
            "ants",
            in_mif,
            out_mif,
            "-mask",
            mask_mif,
            "-bias",
            bias_mif,
        ]
    )
    return out_mif


@pydra.mark.task
def run_n4(t1_nii: str, out_nii: str) -> str:
    run_cmd(["N4BiasFieldCorrection", "-i", t1_nii, "-o", out_nii])
    return out_nii


@pydra.mark.task
def extract_mean_b0(dwi_biascorr_mif: str, out_nii: str) -> str:
    extract = subprocess.Popen(
        ["dwiextract", dwi_biascorr_mif, "-", "-bzero"], stdout=subprocess.PIPE
    )
    subprocess.run(
        ["mrmath", "-", "mean", out_nii, "-axis", "3"], stdin=extract.stdout, check=True
    )
    extract.stdout.close()
    extract.wait()
    return out_nii


@pydra.mark.task
@pydra.mark.annotate({"return": {"b02t1_mat": str}})
def register_b0_to_t1(b0_nii: str, t1_nii: str, out_b0_in_t1: str, out_mat: str):
    run_cmd(
        [
            "flirt",
            "-in",
            b0_nii,
            "-ref",
            t1_nii,
            "-out",
            out_b0_in_t1,
            "-omat",
            out_mat,
            "-dof",
            "6",
        ]
    )
    return out_mat


@pydra.mark.task
@pydra.mark.annotate({"return": {"t1_in_dwi": str, "mrtrix_xfm": str}})
def invert_and_apply_transform(b02t1_mat: str, t1_nii: str, b0_nii: str, tmp_dir: str):
    t12b0_mat = str(Path(tmp_dir) / "T12b0.mat")
    t1_in_dwi = str(Path(tmp_dir) / "T1_n4_in_DWI_space.nii.gz")
    mrtrix_txt = str(Path(tmp_dir) / "struct2diff_mrtrix.txt")
    run_cmd(["convert_xfm", "-omat", t12b0_mat, "-inverse", b02t1_mat])
    run_cmd(
        [
            "flirt",
            "-in",
            t1_nii,
            "-ref",
            b0_nii,
            "-out",
            t1_in_dwi,
            "-init",
            t12b0_mat,
            "-applyxfm",
        ]
    )
    run_cmd(["transformconvert", t12b0_mat, t1_nii, b0_nii, "flirt_import", mrtrix_txt])
    return (t1_in_dwi, mrtrix_txt)


@pydra.mark.task
@pydra.mark.annotate({"return": {"adc": str, "fa": str}})
def compute_tensor_metrics(dwi_biascorr_mif: str, tmp_dir: str):
    tensor_mif = str(Path(tmp_dir) / "tensor.mif.gz")
    adc_mif = str(Path(tmp_dir) / "ADC.mif.gz")
    fa_mif = str(Path(tmp_dir) / "FA.mif.gz")
    adc_nii = str(Path(tmp_dir) / "ADC.nii.gz")
    fa_nii = str(Path(tmp_dir) / "FA.nii.gz")
    run_cmd(["dwi2tensor", dwi_biascorr_mif, tensor_mif])
    run_cmd(["tensor2metric", "-adc", adc_mif, "-fa", fa_mif, tensor_mif])
    run_cmd(["mrconvert", adc_mif, adc_nii])
    run_cmd(["mrconvert", fa_mif, fa_nii])
    return (adc_nii, fa_nii)


@pydra.mark.task
@pydra.mark.annotate(
    {"return": {"dwi_biascorr": str, "t1_in_dwi": str, "adc": str, "fa": str}}
)
def copy_final_outputs(
    dwi_biascorr_mif: str,
    t1_in_dwi: str,
    adc_nii: str,
    fa_nii: str,
    out_dir: str,
    dwi_preproc_name: str,
):
    os.makedirs(out_dir, exist_ok=True)
    dst_dwi = str(Path(out_dir) / dwi_preproc_name)
    dst_t1 = str(Path(out_dir) / "T1_n4_in_DWI_space.nii.gz")
    dst_adc = str(Path(out_dir) / "ADC.nii.gz")
    dst_fa = str(Path(out_dir) / "FA.nii.gz")
    for src, dst in [
        (dwi_biascorr_mif, dst_dwi),
        (t1_in_dwi, dst_t1),
        (adc_nii, dst_adc),
        (fa_nii, dst_fa),
    ]:
        shutil.copy2(src, dst)
        print(f"  Copied: {Path(dst).name}")
    return (dst_dwi, dst_t1, dst_adc, dst_fa)


# =============================================================================
# Per-DWI workflow builder
# =============================================================================


def build_dwi_workflow(plan: dict) -> pydra.Workflow:
    """
    Build a Pydra workflow for a single DWI series from a pre-computed plan.
    T1 N4 bias correction runs in parallel with the DWI chain, converging at
    registration.
    """
    dwi_name = plan["dwi_name"]
    tmp_dir = plan["tmp_dir"]
    out_dir = plan["out_dir"]
    pe_dir = plan["pe_dir"]
    rpe_dir = plan["rpe_dir"]
    preproc_mode = plan["preproc_mode"]
    do_denoise = plan["do_denoise"]
    do_gradcheck = plan["do_gradcheck"]
    readout_time = plan["readout_time"]
    eddy_options = plan["eddy_options"]
    dwi_preproc_name = plan["dwi_preproc_name"]

    dwi_nii = plan["dwi_nii"]
    dwi_json = plan["dwi_json"]
    dwi_bvec = plan["dwi_bvec"]
    dwi_bval = plan["dwi_bval"]
    rpe_nii = plan["rpe_nii"]
    rpe_json = plan["rpe_json"]
    rpe_bvec = plan["rpe_bvec"]
    rpe_bval = plan["rpe_bval"]
    fwd_pe_nii = plan["fwd_pe_nii"]
    fwd_pe_json = plan["fwd_pe_json"]
    fwd_pe_bvec = plan["fwd_pe_bvec"]
    fwd_pe_bval = plan["fwd_pe_bval"]

    safe_name = sanitise_name(dwi_name)
    wf = pydra.Workflow(name=f"dwi_{safe_name}", input_spec=["x"])
    wf.inputs.x = 1

    # ------------------------------------------------------------------
    # T1 branch (parallel)
    # ------------------------------------------------------------------
    t1_conv_dir = str(Path(tmp_dir) / "t1_nii")
    wf.add(
        convert_dicoms(
            name="convert_t1",
            dicom_dir=plan["t1_dir"],
            out_dir=t1_conv_dir,
        )
    )
    wf.add(
        run_n4(
            name="n4_t1",
            t1_nii=wf.convert_t1.lzout.nii,
            out_nii=str(Path(tmp_dir) / "T1_n4.nii.gz"),
        )
    )

    # ------------------------------------------------------------------
    # DWI: mrconvert to MIF (with optional gradcheck)
    # ------------------------------------------------------------------
    dwi_raw_mif = str(Path(tmp_dir) / f"DWI_raw_{pe_dir}.mif.gz")

    if do_gradcheck:
        wf.add(
            convert_to_mif_initial(
                name="dwi_to_mif_init",
                nii=dwi_nii,
                json_file=dwi_json,
                bvec=dwi_bvec,
                bval=dwi_bval,
                out_path=str(Path(tmp_dir) / f"DWI_raw_{pe_dir}_init.mif.gz"),
            )
        )
        wf.add(
            run_gradcheck(
                name="gradcheck_dwi",
                mif_path=wf.dwi_to_mif_init.lzout.out,
                out_dir=tmp_dir,
                prefix="dwi",
            )
        )
        wf.add(
            convert_to_mif_final(
                name="dwi_to_mif",
                nii=dwi_nii,
                json_file=dwi_json,
                bvec=wf.gradcheck_dwi.lzout.bvec,
                bval=wf.gradcheck_dwi.lzout.bval,
                out_path=dwi_raw_mif,
            )
        )

        if preproc_mode == "rpe_all" and rpe_nii:
            rpe_raw_mif = str(Path(tmp_dir) / f"DWI_raw_{rpe_dir}.mif.gz")
            wf.add(
                convert_to_mif_initial(
                    name="rpe_to_mif_init",
                    nii=rpe_nii,
                    json_file=rpe_json,
                    bvec=rpe_bvec,
                    bval=rpe_bval,
                    out_path=str(Path(tmp_dir) / f"DWI_raw_{rpe_dir}_init.mif.gz"),
                )
            )
            wf.add(
                run_gradcheck(
                    name="gradcheck_rpe",
                    mif_path=wf.rpe_to_mif_init.lzout.out,
                    out_dir=tmp_dir,
                    prefix="rpe",
                )
            )
            wf.add(
                convert_to_mif_final(
                    name="rpe_to_mif",
                    nii=rpe_nii,
                    json_file=rpe_json,
                    bvec=wf.gradcheck_rpe.lzout.bvec,
                    bval=wf.gradcheck_rpe.lzout.bval,
                    out_path=rpe_raw_mif,
                )
            )
            rpe_mif_out = wf.rpe_to_mif.lzout.out

        elif fwd_pe_nii and preproc_mode in ("rpe_pair", "rpe_split"):
            wf.add(
                convert_to_mif_initial(
                    name="fwd_to_mif_init",
                    nii=fwd_pe_nii,
                    json_file=fwd_pe_json,
                    bvec=fwd_pe_bvec,
                    bval=fwd_pe_bval,
                    out_path=str(Path(tmp_dir) / f"DWI_ref_{pe_dir}_init.mif.gz"),
                )
            )
            wf.add(
                run_gradcheck(
                    name="gradcheck_fwd",
                    mif_path=wf.fwd_to_mif_init.lzout.out,
                    out_dir=tmp_dir,
                    prefix="fwd",
                )
            )
            fwd_bvec_out = wf.gradcheck_fwd.lzout.bvec
            fwd_bval_out = wf.gradcheck_fwd.lzout.bval
        else:
            fwd_bvec_out = fwd_pe_bvec
            fwd_bval_out = fwd_pe_bval

        dwi_mif_out = wf.dwi_to_mif.lzout.out

    else:
        wf.add(
            convert_to_mif_final(
                name="dwi_to_mif",
                nii=dwi_nii,
                json_file=dwi_json,
                bvec=dwi_bvec,
                bval=dwi_bval,
                out_path=dwi_raw_mif,
            )
        )
        dwi_mif_out = wf.dwi_to_mif.lzout.out
        fwd_bvec_out = fwd_pe_bvec
        fwd_bval_out = fwd_pe_bval

        if preproc_mode == "rpe_all" and rpe_nii:
            rpe_raw_mif = str(Path(tmp_dir) / f"DWI_raw_{rpe_dir}.mif.gz")
            wf.add(
                convert_to_mif_final(
                    name="rpe_to_mif",
                    nii=rpe_nii,
                    json_file=rpe_json,
                    bvec=rpe_bvec,
                    bval=rpe_bval,
                    out_path=rpe_raw_mif,
                )
            )
            rpe_mif_out = wf.rpe_to_mif.lzout.out

    # ------------------------------------------------------------------
    # Denoise + Gibbs (applied to AP and PA separately for rpe_all)
    # ------------------------------------------------------------------
    if do_denoise:
        wf.add(
            run_dwidenoise(
                name="denoise_dwi",
                in_mif=dwi_mif_out,
                out_mif=str(Path(tmp_dir) / f"DWI_raw_{pe_dir}_denoise.mif.gz"),
            )
        )
        wf.add(
            run_mrdegibbs(
                name="degibbs_dwi",
                in_mif=wf.denoise_dwi.lzout.out,
                out_mif=str(Path(tmp_dir) / f"DWI_raw_{pe_dir}_denoise_gibbs.mif.gz"),
            )
        )
        dwi_for_preproc = wf.degibbs_dwi.lzout.out

        if preproc_mode == "rpe_all" and rpe_nii:
            wf.add(
                run_dwidenoise(
                    name="denoise_rpe",
                    in_mif=rpe_mif_out,
                    out_mif=str(Path(tmp_dir) / f"DWI_raw_{rpe_dir}_denoise.mif.gz"),
                )
            )
            wf.add(
                run_mrdegibbs(
                    name="degibbs_rpe",
                    in_mif=wf.denoise_rpe.lzout.out,
                    out_mif=str(
                        Path(tmp_dir) / f"DWI_raw_{rpe_dir}_denoise_gibbs.mif.gz"
                    ),
                )
            )
            rpe_for_preproc = wf.degibbs_rpe.lzout.out
        elif preproc_mode == "rpe_all":
            rpe_for_preproc = rpe_mif_out
    else:
        dwi_for_preproc = dwi_mif_out
        if preproc_mode == "rpe_all" and rpe_nii:
            rpe_for_preproc = rpe_mif_out

    # ------------------------------------------------------------------
    # Concatenate AP + PA for rpe_all
    # ------------------------------------------------------------------
    if preproc_mode == "rpe_all" and rpe_nii:
        wf.add(
            concatenate_ap_pa(
                name="concat_ap_pa",
                ap_mif=dwi_for_preproc,
                pa_mif=rpe_for_preproc,
                out_mif=str(Path(tmp_dir) / "DWI_AP_PA_concat.mif.gz"),
            )
        )
        dwi_input_for_preproc = wf.concat_ap_pa.lzout.out
    else:
        dwi_input_for_preproc = dwi_for_preproc

    # ------------------------------------------------------------------
    # se_epi pair (rpe_pair / rpe_split only)
    # ------------------------------------------------------------------
    wf.add(
        build_se_epi(
            name="se_epi",
            dwi_mif=dwi_for_preproc,
            rpe_nii=rpe_nii or "",
            rpe_json=rpe_json,
            rpe_bvec=rpe_bvec,
            rpe_bval=rpe_bval,
            fwd_pe_nii=fwd_pe_nii or "",
            fwd_pe_json=fwd_pe_json,
            fwd_pe_bvec=fwd_bvec_out,
            fwd_pe_bval=fwd_bval_out,
            pe_dir=pe_dir,
            rpe_dir=rpe_dir,
            preproc_mode=preproc_mode,
            tmp_dir=tmp_dir,
        )
    )

    # ------------------------------------------------------------------
    # dwifslpreproc
    # ------------------------------------------------------------------
    wf.add(
        run_dwifslpreproc(
            name="fslpreproc",
            dwi_mif=dwi_input_for_preproc,
            out_mif=str(Path(tmp_dir) / "DWI_preproc.mif.gz"),
            pe_dir=pe_dir,
            preproc_mode=preproc_mode,
            se_epi=wf.se_epi.lzout.out,
            readout_time=readout_time,
            eddy_options=eddy_options,
        )
    )

    # ------------------------------------------------------------------
    # Mask + DWI bias correction
    # ------------------------------------------------------------------
    wf.add(
        run_dwi2mask(
            name="dwi_mask",
            in_mif=wf.fslpreproc.lzout.out,
            out_mif=str(Path(tmp_dir) / "DWI_preproc_mask.mif.gz"),
        )
    )
    wf.add(
        run_dwibiascorrect(
            name="dwi_biascorr",
            in_mif=wf.fslpreproc.lzout.out,
            out_mif=str(Path(tmp_dir) / dwi_preproc_name),
            mask_mif=wf.dwi_mask.lzout.out,
            bias_mif=str(Path(tmp_dir) / "DWI_preproc_bias.mif.gz"),
        )
    )

    # ------------------------------------------------------------------
    # Mean b0 extraction
    # ------------------------------------------------------------------
    wf.add(
        extract_mean_b0(
            name="mean_b0",
            dwi_biascorr_mif=wf.dwi_biascorr.lzout.out,
            out_nii=str(Path(tmp_dir) / "bzero_f.nii.gz"),
        )
    )

    # ------------------------------------------------------------------
    # Registration (T1 and DWI branches converge here)
    # ------------------------------------------------------------------
    wf.add(
        register_b0_to_t1(
            name="register",
            b0_nii=wf.mean_b0.lzout.out,
            t1_nii=wf.n4_t1.lzout.out,
            out_b0_in_t1=str(Path(tmp_dir) / "b0_to_T1.nii.gz"),
            out_mat=str(Path(tmp_dir) / "b02T1.mat"),
        )
    )
    wf.add(
        invert_and_apply_transform(
            name="invert_xfm",
            b02t1_mat=wf.register.lzout.b02t1_mat,
            t1_nii=wf.n4_t1.lzout.out,
            b0_nii=wf.mean_b0.lzout.out,
            tmp_dir=tmp_dir,
        )
    )

    # ------------------------------------------------------------------
    # Tensor metrics
    # ------------------------------------------------------------------
    wf.add(
        compute_tensor_metrics(
            name="tensor",
            dwi_biascorr_mif=wf.dwi_biascorr.lzout.out,
            tmp_dir=tmp_dir,
        )
    )

    # ------------------------------------------------------------------
    # Copy final outputs
    # ------------------------------------------------------------------
    wf.add(
        copy_final_outputs(
            name="copy_outputs",
            dwi_biascorr_mif=wf.dwi_biascorr.lzout.out,
            t1_in_dwi=wf.invert_xfm.lzout.t1_in_dwi,
            adc_nii=wf.tensor.lzout.adc,
            fa_nii=wf.tensor.lzout.fa,
            out_dir=out_dir,
            dwi_preproc_name=dwi_preproc_name,
        )
    )

    wf.set_output([("outputs", wf.copy_outputs.lzout.dwi_biascorr)])
    return wf


# =============================================================================
# Top-level pipeline
# =============================================================================


def run_pipeline(cfg: dict):
    scans_dir = cfg["scans_dir"]
    output_dir = cfg.get("output_dir", str(Path(scans_dir).name))
    cfg["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scans directory:  {scans_dir}")
    print(f"Denoise/Degibbs:  {cfg.get('denoise_degibbs', False)}")
    print(f"Gradcheck:        {cfg.get('gradcheck', False)}")
    print(f"Output:           {Path(output_dir).resolve()}")

    # Step 1: classify directories (provisional)
    dirs = scan_directory(scans_dir)
    print(f"\nFound:")
    print(f"  {len(dirs['t1_dirs'])} T1 series")
    print(f"  {len(dirs['candidate_dwi'])} candidate DWI series (before filtering)")
    if dirs["ignored"]:
        print(f"  {len(dirs['ignored'])} ignored (non-DWI)")

    # Step 2: convert all candidate DICOMs
    print("\nConverting DICOMs for all candidate series...")
    conversions = convert_all_candidates(dirs["candidate_dwi"], output_dir)

    # Steps 3-6: validate bvec/bval, check b-values, finalise classification
    classified = classify_candidates(dirs["candidate_dwi"], conversions)

    # Steps 7: detect AP/PA pairs (across dwi_dirs and pending series)
    print("\nDetecting AP/PA pairs...")
    # Pass existing fwd/rpe dirs into match so they are preserved
    conversions["fwd_pe_dirs"] = classified["fwd_pe_dirs"]
    conversions["rpe_dirs"] = classified["rpe_dirs"]
    dwi_dirs, fwd_pe_dirs, rpe_dirs, rpe_all_map, tie_warnings = match_ap_pa_pairs(
        classified["dwi_dirs"],
        classified["pending_fwd"],
        classified["pending_rpe"],
        conversions,
    )

    print(f"\nAfter classification:")
    print(f"  {len(dwi_dirs)} DWI series to process")
    print(f"  {len(fwd_pe_dirs)} forward PE images")
    print(f"  {len(rpe_dirs)} reverse PE images")
    print(f"  {len(rpe_all_map)} rpe_all pairs detected")
    if classified["skipped"]:
        print(f"  {len(classified['skipped'])} series skipped")

    if not dwi_dirs:
        raise ValueError("No processable DWI series found after filtering.")

    # Steps 8-11: plan each workflow
    print("\nPlanning workflows...")
    plans = []
    for dwi_dir in dwi_dirs:
        plan = plan_workflow(
            dwi_dir=dwi_dir,
            t1_dirs=dirs["t1_dirs"],
            fwd_pe_dirs=fwd_pe_dirs,
            rpe_dirs=rpe_dirs,
            rpe_all_map=rpe_all_map,
            tie_warnings=tie_warnings,
            conversions=conversions,
            cfg=cfg,
        )
        plans.append(plan)

    # Step 12: print plan and pause
    print_plan(plans, classified["skipped"])

    # Build and run workflows
    sub_workflows = []
    for plan in plans:
        wf = build_dwi_workflow(plan)
        sub_workflows.append(wf)

    top = pydra.Workflow(name="dwi_pipeline", input_spec=["x"])
    top.inputs.x = 1
    for wf in sub_workflows:
        top.add(wf)

    top.set_output(
        [
            (f"out_{sanitise_name(wf.name)}", getattr(top, wf.name).lzout.outputs)
            for wf in sub_workflows
        ]
    )

    with pydra.Submitter(plugin="cf") as sub:
        sub(top)

    results = top.result()
    print("\n=== All done ===")
    print(f"Outputs in: {Path(output_dir).resolve()}")
    return results


# =============================================================================
# CLI
# =============================================================================


def load_config(args) -> dict:
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    if args.scans_dir:
        cfg["scans_dir"] = args.scans_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.denoise_degibbs:
        cfg["denoise_degibbs"] = True
    if args.gradcheck:
        cfg["gradcheck"] = True
    if args.readout_time is not None:
        cfg["readout_time"] = args.readout_time
    if args.eddy_options is not None:
        cfg["eddy_options"] = args.eddy_options
    if "scans_dir" not in cfg:
        raise ValueError("scans_dir must be provided via --scans-dir or config YAML")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="DWI processing and tensor metrics pipeline"
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--scans-dir", type=str, help="Path to scans directory")
    parser.add_argument("--output-dir", type=str, help="Path to output directory")
    parser.add_argument("--denoise-degibbs", action="store_true", default=None)
    parser.add_argument("--gradcheck", action="store_true", default=None)
    parser.add_argument("--readout-time", type=float, default=None)
    parser.add_argument("--eddy-options", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
