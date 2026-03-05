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
    """Return number of volumes using mrinfo -ndim."""
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


def detect_pe_direction(folder_name: str) -> tuple:
    """Return (pe_dir, rpe_dir) from a folder name."""
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
        f"Could not determine PE direction from: {folder_name}\n"
        f"Expected one of: AP, PA, LR, RL, SI, IS (or underscore variants)"
    )


def is_pe_direction(name: str, directions: list) -> bool:
    """Check if a folder name contains any of the given PE direction tags."""
    for d in directions:
        if re.search(rf"_{d}(_|$)", name, re.IGNORECASE):
            return True
        if re.search(rf"_{d[0]}_{d[1]}(_|$)", name, re.IGNORECASE):
            return True
    return False


# =============================================================================
# Directory scanning and workflow planning
# =============================================================================

ALL_PE_DIRS = [
    "AP",
    "PA",
    "LR",
    "RL",
    "SI",
    "IS",
    "A_P",
    "P_A",
    "L_R",
    "R_L",
    "S_I",
    "I_S",
]
FWD_DIRS = ["AP", "LR", "SI", "A_P", "L_R", "S_I"]
RPE_DIRS = ["PA", "RL", "IS", "P_A", "R_L", "I_S"]


def scan_directory(scans_dir: str) -> dict:
    """
    Classify all subdirectories in scans_dir into:
      - t1_dirs: list of T1 directories (sorted by series number)
      - dwi_dirs: list of main DWI directories
      - fwd_pe_dirs: list of forward PE (b0 or full) directories
      - rpe_dirs: list of reverse PE (b0 or full) directories
      - ignored: list of skipped directories (ADC, FA scanner maps)

    PE direction tags are matched anywhere in the folder name, not just at end.
    """
    scans_path = Path(scans_dir)
    t1_dirs = []
    dwi_dirs = []
    fwd_pe_dirs = []
    rpe_dirs = []
    ignored = []

    for d in sorted(scans_path.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        # T1: contains "t1" (case-insensitive)
        if re.search(r"t1", name, re.IGNORECASE):
            t1_dirs.append(str(d))
            continue

        # Skip scanner-derived maps
        if re.search(r"_(ADC|FA)$", name, re.IGNORECASE):
            ignored.append(str(d))
            continue

        # Only classify further if this looks like a DWI acquisition
        if not re.search(r"(_diff_|_DWI_)", name, re.IGNORECASE):
            ignored.append(str(d))
            continue

        # Distinguish main DWI from PE pair images.
        # PE pair images are identified by having a PE direction tag as the
        # FINAL meaningful token (nothing substantial after it).
        # Main DWI acquisitions have additional parameter suffixes after the
        # PE direction tag (e.g. _BW3720, _ESpt57, _BIPOLAR).
        #
        # Strategy: check if the PE direction tag appears with only short/numeric
        # suffixes after it (<=15 chars). If so it's a PE pair image; otherwise
        # it's a main DWI.
        def has_terminal_pe(n, directions):
            """
            Returns True if a PE direction tag appears as the last meaningful
            token in the folder name, i.e. nothing follows it, or only a
            short purely alphanumeric suffix (no underscores — one token only).
            This prevents '_AP_ORIG_P_A' from matching as a forward PE image.
            """
            for d_ in directions:
                # Bare direction tag at end, with optional single short alphanumeric token
                for pat in [rf"_{d_}$", rf"_{d_}_[A-Za-z0-9]{{1,10}}$"]:
                    if re.search(pat, n, re.IGNORECASE):
                        return True
                # Underscore-separated direction (e.g. A_P, P_A) at end
                if len(d_) == 3 and d_[1] == "_":
                    for pat in [
                        rf"_{d_[0]}_{d_[2]}$",
                        rf"_{d_[0]}_{d_[2]}_[A-Za-z0-9]{{1,10}}$",
                    ]:
                        if re.search(pat, n, re.IGNORECASE):
                            return True
            return False

        if has_terminal_pe(name, FWD_DIRS):
            fwd_pe_dirs.append(str(d))
            continue

        if has_terminal_pe(name, RPE_DIRS):
            rpe_dirs.append(str(d))
            continue

        # Has a PE direction tag mid-name with substantial suffixes: main DWI
        dwi_dirs.append(str(d))

    if not t1_dirs:
        raise ValueError(f"Could not identify any T1 directory in {scans_dir}")
    if not dwi_dirs:
        raise ValueError(f"Could not identify any DWI directories in {scans_dir}")

    # Detect AP/PA (rpe_all) pairs among the main DWI dirs
    dwi_dirs, rpe_all_map = match_ap_pa_pairs(dwi_dirs)

    return {
        "t1_dirs": t1_dirs,
        "dwi_dirs": sorted(dwi_dirs),
        "fwd_pe_dirs": fwd_pe_dirs,
        "rpe_dirs": rpe_dirs,
        "rpe_all_map": rpe_all_map,  # {fwd_path: rpe_path} for rpe_all pairs
        "ignored": ignored,
    }


def get_acq_stem(folder_name: str) -> str:
    """
    Extract the acquisition stem for AP/PA matching by:
    1. Stripping the leading series number
    2. Removing the PE direction tag and everything after it
    e.g. '12-ep2d_diff__FREE30DIR_b1000_x7b0_Ghost_R_AP_ESpt54_BW2264_BIPOLAR'
      -> 'ep2d_diff__FREE30DIR_b1000_x7b0_Ghost_R'
    """
    name = strip_series_number(folder_name)
    all_dirs = [
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
    for d in all_dirs:
        m = re.search(rf"_({re.escape(d)})(_|$)", name, re.IGNORECASE)
        if m:
            return name[: m.start()]
    return name


def match_ap_pa_pairs(dwi_dirs: list) -> tuple:
    """
    Scan dwi_dirs for AP/PA (or LR/RL, SI/IS) pairs sharing the same
    acquisition stem. The forward-direction series stays in dwi_dirs as
    the main DWI; the reverse-direction series is returned in a
    rpe_all_map dict: {fwd_dir_path: rpe_dir_path}.

    Returns (filtered_dwi_dirs, rpe_all_map).
    """
    stem_map = {}
    for d in dwi_dirs:
        name = Path(d).name
        try:
            pe, _ = detect_pe_direction(name)
        except ValueError:
            pe = None
        stem = get_acq_stem(name)
        stem_map.setdefault(stem, []).append((d, pe))

    fwd_dirs = []
    rpe_all_map = {}

    fwd_set = {"AP", "LR", "SI"}
    rpe_set = {"PA", "RL", "IS"}

    for stem, entries in stem_map.items():
        if len(entries) == 1:
            fwd_dirs.append(entries[0][0])
            continue

        fwds = [(p, pe) for p, pe in entries if pe in fwd_set]
        rpes = [(p, pe) for p, pe in entries if pe in rpe_set]

        if fwds and rpes:
            for fwd_path, fwd_pe in fwds:
                fwd_num = get_series_number(Path(fwd_path).name)
                best_rpe = min(
                    rpes,
                    key=lambda x: abs(get_series_number(Path(x[0]).name) - fwd_num),
                )
                rpe_all_map[fwd_path] = best_rpe[0]
                fwd_dirs.append(fwd_path)
            matched_rpes = set(rpe_all_map.values())
            for rpe_path, _ in rpes:
                if rpe_path not in matched_rpes:
                    fwd_dirs.append(rpe_path)
        else:
            for p, _ in entries:
                fwd_dirs.append(p)

    return fwd_dirs, rpe_all_map


def assign_t1(dwi_name: str, t1_dirs: list) -> str:
    """
    Assign the correct T1 to a DWI series.
    Rule: the T1 with the largest series number that is still less than
    the DWI series number. Falls back to the first T1 if none precedes it.
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


def find_best_pe_match(dwi_name: str, pe_dirs: list) -> str:
    """
    Match a PE directory to a DWI by longest common prefix of acquisition
    name (series number stripped). Falls back to first available if no
    strong match (>10 chars).
    """
    dwi_stem = strip_series_number(dwi_name)
    best_dir = None
    best_len = 0
    for d in pe_dirs:
        pe_stem = strip_series_number(Path(d).name)
        length = common_prefix_len(dwi_stem, pe_stem)
        if length > best_len:
            best_len = length
            best_dir = d
    if best_len > 10 and best_dir is not None:
        return best_dir
    return pe_dirs[0] if pe_dirs else None


def plan_workflow(
    dwi_dir: str,
    t1_dirs: list,
    fwd_pe_dirs: list,
    rpe_dirs: list,
    rpe_all_map: dict,
    cfg: dict,
) -> dict:
    """
    Plan the full workflow for a single DWI series.
    Performs DICOM conversion to count volumes and determine preproc mode.
    Returns a plan dict describing every step to be executed.
    """
    dwi_name = Path(dwi_dir).name
    pe_dir, rpe_dir = detect_pe_direction(dwi_name)

    t1_dir = assign_t1(dwi_name, t1_dirs)

    # Check if this DWI has a full-volume PA partner (rpe_all)
    rpe_all_partner = rpe_all_map.get(dwi_dir)

    # For rpe_all pairs, use the partner as the RPE; ignore b0 PE images
    if rpe_all_partner:
        fwd_pe_dir = None  # b0 FWD PE not needed for rpe_all
        rpe_dir_path = rpe_all_partner
    else:
        fwd_pe_dir = find_best_pe_match(dwi_name, fwd_pe_dirs) if fwd_pe_dirs else None
        rpe_dir_path = find_best_pe_match(dwi_name, rpe_dirs) if rpe_dirs else None

    do_denoise = cfg.get("denoise_degibbs", False)
    do_gradcheck = cfg.get("gradcheck", False)
    readout_time_override = cfg.get("readout_time", None)
    eddy_options = cfg.get("eddy_options", " --slm=linear")

    # Determine preproc mode from volume counts
    # We need to do a quick dcm2niix to NIfTI to count volumes.
    # This is done in a scratch dir that will be reused by the main pipeline.
    out_dir = str(Path(cfg["output_dir"]) / dwi_name)
    tmp_dir = str(Path(out_dir) / "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Convert DWI and RPE DICOMs temporarily to count volumes
    dwi_conv_dir = str(Path(tmp_dir) / "dwi_nii")
    rpe_conv_dir = str(Path(tmp_dir) / "rpe_nii") if rpe_dir_path else None
    fwd_conv_dir = str(Path(tmp_dir) / "fwd_pe_nii") if fwd_pe_dir else None

    os.makedirs(dwi_conv_dir, exist_ok=True)
    subprocess.run(
        ["dcm2niix", "-o", dwi_conv_dir, "-f", "%p", "-z", "y", dwi_dir],
        check=True,
        capture_output=True,
    )

    dwi_niis = sorted(Path(dwi_conv_dir).glob("*.nii.gz"))
    if not dwi_niis:
        raise FileNotFoundError(f"No NIfTI produced from DWI DICOM: {dwi_dir}")
    dwi_nii = str(dwi_niis[0])
    dwi_nvols = get_nvols(dwi_nii)

    rpe_nii = None
    rpe_nvols = 0
    if rpe_dir_path:
        os.makedirs(rpe_conv_dir, exist_ok=True)
        subprocess.run(
            ["dcm2niix", "-o", rpe_conv_dir, "-f", "%p", "-z", "y", rpe_dir_path],
            check=True,
            capture_output=True,
        )
        rpe_niis = sorted(Path(rpe_conv_dir).glob("*.nii.gz"))
        if rpe_niis:
            rpe_nii = str(rpe_niis[0])
            rpe_nvols = get_nvols(rpe_nii)

    fwd_pe_nii = None
    if fwd_pe_dir:
        os.makedirs(fwd_conv_dir, exist_ok=True)
        subprocess.run(
            ["dcm2niix", "-o", fwd_conv_dir, "-f", "%p", "-z", "y", fwd_pe_dir],
            check=True,
            capture_output=True,
        )
        fwd_niis = sorted(Path(fwd_conv_dir).glob("*.nii.gz"))
        if fwd_niis:
            fwd_pe_nii = str(fwd_niis[0])

    # Determine preproc mode
    if rpe_nii is None:
        preproc_mode = "rpe_none"
    elif rpe_nvols == dwi_nvols:
        preproc_mode = "rpe_all"
    elif rpe_nvols > 1 and fwd_pe_nii is not None:
        preproc_mode = "rpe_split"
    else:
        preproc_mode = "rpe_pair"

    # Resolve readout time
    dwi_json = dwi_nii.replace(".nii.gz", ".json")
    if readout_time_override is not None:
        readout_time = float(readout_time_override)
        readout_time_source = "config override"
    elif Path(dwi_json).exists():
        readout_time = get_readout_time(dwi_json)
        readout_time_source = f"JSON ({dwi_json})"
    else:
        readout_time = 0.0342002
        readout_time_source = "fallback (no JSON found)"

    # Output filename reflects steps applied
    if do_denoise:
        dwi_preproc_name = "DWI_denoise_gibbs_preproc_biascorr.mif.gz"
    else:
        dwi_preproc_name = "DWI_preproc_biascorr.mif.gz"

    return {
        "dwi_name": dwi_name,
        "dwi_dir": dwi_dir,
        "t1_dir": t1_dir,
        "fwd_pe_dir": fwd_pe_dir,
        "rpe_dir_path": rpe_dir_path,
        "pe_dir": pe_dir,
        "rpe_dir": rpe_dir,
        "dwi_nii": dwi_nii,
        "dwi_nvols": dwi_nvols,
        "rpe_nii": rpe_nii,
        "rpe_nvols": rpe_nvols,
        "fwd_pe_nii": fwd_pe_nii,
        "preproc_mode": preproc_mode,
        "readout_time": readout_time,
        "readout_time_src": readout_time_source,
        "do_denoise": do_denoise,
        "do_gradcheck": do_gradcheck,
        "eddy_options": eddy_options,
        "dwi_preproc_name": dwi_preproc_name,
        "out_dir": out_dir,
        "tmp_dir": tmp_dir,
        "dwi_conv_dir": dwi_conv_dir,
        "rpe_conv_dir": rpe_conv_dir,
        "fwd_conv_dir": fwd_conv_dir,
    }


def print_plan(plans: list):
    """Print a human-readable workflow summary before execution begins."""
    print("\n" + "=" * 70)
    print("WORKFLOW PLAN")
    print("=" * 70)
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
    """
    Step 1 of gradcheck flow: mrconvert NIfTI -> MIF using original gradients.
    Gradients are embedded so dwigradcheck can operate on the MIF directly.
    """
    run_cmd(
        ["mrconvert", nii, out_path, "-json_import", json_file, "-fslgrad", bvec, bval]
    )
    return out_path


@pydra.mark.task
@pydra.mark.annotate({"return": {"bvec": str, "bval": str}})
def run_gradcheck(mif_path: str, out_dir: str, prefix: str):
    """
    Run dwigradcheck on a MIF (gradients already embedded).
    Exports corrected bvec/bval.
    """
    out_bvec = str(Path(out_dir) / f"{prefix}_corrected.bvec")
    out_bval = str(Path(out_dir) / f"{prefix}_corrected.bval")
    run_cmd(["dwigradcheck", mif_path, "-export_grad_fsl", out_bvec, out_bval])
    return (out_bvec, out_bval)


@pydra.mark.task
def convert_to_mif_final(
    nii: str, json_file: str, bvec: str, bval: str, out_path: str
) -> str:
    """
    Step 2 of gradcheck flow: re-run mrconvert using corrected gradients.
    Also used as the only mrconvert step when gradcheck is disabled.
    """
    run_cmd(
        ["mrconvert", nii, out_path, "-json_import", json_file, "-fslgrad", bvec, bval]
    )
    # Verify PE table embedded
    result = subprocess.run(
        ["mrinfo", out_path, "-petable"], capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"  Warning: PE table not found in MIF header: {out_path}")
        print(
            f"           Check {json_file} contains PhaseEncodingDirection "
            f"and TotalReadoutTime"
        )
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
    """Concatenate AP and PA DWI series for rpe_all (AP first, PA last)."""
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
    """
    Build se_epi pair for rpe_pair / rpe_split.
    Returns path to bzero_pair.mif.gz, or empty string if not needed.
    """
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
        print("  No forward PE image -- computing mean b0 from DWI...")
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
    """
    Run dwifslpreproc.
    For rpe_all: dwi_mif must already be the AP+PA concatenated image.
    For rpe_pair/rpe_split: se_epi is the bzero_pair image.
    """
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
        cmd += ["-se_epi", se_epi, "-readout_time", str(readout_time)]
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
    """Extract mean b0 from bias-corrected DWI."""
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
    """Register mean b0 to T1 using FLIRT (6 DOF)."""
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
    """Invert b0->T1 transform and resample T1 into DWI space."""
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
    """Run dwi2tensor and tensor2metric, output NIfTI ADC and FA."""
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
    """Copy the four key outputs from tmp/ to the output directory."""
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
    Build a Pydra workflow for a single DWI series based on a pre-computed plan.
    Parallelism:
      - T1 N4 bias correction runs in parallel with the DWI processing chain
      - Both converge at registration
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

    fwd_pe_dir = plan["fwd_pe_dir"]
    rpe_dir_path = plan["rpe_dir_path"]
    dwi_conv_dir = plan["dwi_conv_dir"]
    rpe_conv_dir = plan["rpe_conv_dir"]
    fwd_conv_dir = plan["fwd_conv_dir"]

    safe_name = sanitise_name(dwi_name)
    wf = pydra.Workflow(name=f"dwi_{safe_name}", input_spec=["x"])
    wf.inputs.x = 1

    # ------------------------------------------------------------------
    # T1 conversion + N4 (parallel branch)
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
    # DWI: use already-converted NIfTI from planning stage
    # (dcm2niix already ran on DWI, RPE, FWD PE during plan_workflow)
    # ------------------------------------------------------------------
    dwi_nii = plan["dwi_nii"]
    dwi_json = dwi_nii.replace(".nii.gz", ".json")
    dwi_bvec = dwi_nii.replace(".nii.gz", ".bvec")
    dwi_bval = dwi_nii.replace(".nii.gz", ".bval")

    rpe_nii = plan["rpe_nii"]
    rpe_json = rpe_nii.replace(".nii.gz", ".json") if rpe_nii else ""
    rpe_bvec = rpe_nii.replace(".nii.gz", ".bvec") if rpe_nii else ""
    rpe_bval = rpe_nii.replace(".nii.gz", ".bval") if rpe_nii else ""

    fwd_pe_nii = plan["fwd_pe_nii"]
    fwd_pe_json = fwd_pe_nii.replace(".nii.gz", ".json") if fwd_pe_nii else ""
    fwd_pe_bvec = fwd_pe_nii.replace(".nii.gz", ".bvec") if fwd_pe_nii else ""
    fwd_pe_bval = fwd_pe_nii.replace(".nii.gz", ".bval") if fwd_pe_nii else ""

    # ------------------------------------------------------------------
    # Gradcheck (change #8: operates on MIF, not NIfTI)
    # For rpe_all: run gradcheck on both AP and PA independently
    # ------------------------------------------------------------------
    dwi_raw_mif = str(Path(tmp_dir) / f"DWI_raw_{pe_dir}.mif.gz")

    if do_gradcheck:
        # Step 1: initial mrconvert to embed gradients
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
        # Step 2: gradcheck on MIF
        wf.add(
            run_gradcheck(
                name="gradcheck_dwi",
                mif_path=wf.dwi_to_mif_init.lzout.out,
                out_dir=tmp_dir,
                prefix="dwi",
            )
        )
        # Step 3: final mrconvert with corrected gradients
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

        # For rpe_all: gradcheck PA independently
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

        # Gradcheck for FWD PE (rpe_pair / rpe_split)
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
        # No gradcheck: single mrconvert step
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
    # Denoise + Gibbs (change #6: for rpe_all, applied to AP and PA separately)
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
    # Concatenate AP + PA for rpe_all (change #5)
    # ------------------------------------------------------------------
    if preproc_mode == "rpe_all" and rpe_nii:
        wf.add(
            concatenate_ap_pa(
                name="concat_ap_pa",
                ap_mif=dwi_for_preproc,
                pa_mif=rpe_for_preproc,
                out_mif=str(Path(tmp_dir) / f"DWI_AP_PA_concat.mif.gz"),
            )
        )
        dwi_input_for_preproc = wf.concat_ap_pa.lzout.out
    else:
        dwi_input_for_preproc = dwi_for_preproc

    # ------------------------------------------------------------------
    # Build se_epi pair (rpe_pair / rpe_split only)
    # b0-only images are ignored for rpe_all (change #3)
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
    # Extract mean b0 for registration
    # ------------------------------------------------------------------
    wf.add(
        extract_mean_b0(
            name="mean_b0",
            dwi_biascorr_mif=wf.dwi_biascorr.lzout.out,
            out_nii=str(Path(tmp_dir) / "bzero_f.nii.gz"),
        )
    )

    # ------------------------------------------------------------------
    # Registration (converges DWI and T1 branches)
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

    # Scan and classify directories
    dirs = scan_directory(scans_dir)
    print(f"\nFound:")
    print(f"  {len(dirs['t1_dirs'])} T1 series")
    print(f"  {len(dirs['dwi_dirs'])} DWI series")
    print(f"  {len(dirs['fwd_pe_dirs'])} forward PE images")
    print(f"  {len(dirs['rpe_dirs'])} reverse PE images")
    if dirs["ignored"]:
        print(f"  {len(dirs['ignored'])} ignored (ADC/FA scanner maps)")

    # Plan workflows (includes DICOM conversion for volume counting)
    print("\nPlanning workflows (converting DICOMs for volume counting)...")
    plans = []
    for dwi_dir in dirs["dwi_dirs"]:
        plan = plan_workflow(
            dwi_dir=dwi_dir,
            t1_dirs=dirs["t1_dirs"],
            fwd_pe_dirs=dirs["fwd_pe_dirs"],
            rpe_dirs=dirs["rpe_dirs"],
            rpe_all_map=dirs["rpe_all_map"],
            cfg=cfg,
        )
        plans.append(plan)

    # Print summary and pause for user confirmation
    print_plan(plans)

    # Build one sub-workflow per DWI series
    sub_workflows = []
    for plan in plans:
        wf = build_dwi_workflow(plan)
        sub_workflows.append(wf)

    # Top-level workflow — all DWI sub-workflows run in parallel
    top = pydra.Workflow(name="dwi_pipeline", input_spec=["x"])
    top.inputs.x = 1
    for wf in sub_workflows:
        top.add(wf)

    # Use sanitised names for output attributes (fixes the SyntaxError)
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
    if args.denoise_degibbs:
        cfg["denoise_degibbs"] = True
    if args.gradcheck:
        cfg["gradcheck"] = True
    if args.readout_time is not None:
        cfg["readout_time"] = args.readout_time
    if args.eddy_options is not None:
        cfg["eddy_options"] = args.eddy_options
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
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
    parser.add_argument(
        "--denoise-degibbs",
        action="store_true",
        default=None,
        help="Apply dwidenoise and mrdegibbs",
    )
    parser.add_argument(
        "--gradcheck", action="store_true", default=None, help="Apply dwigradcheck"
    )
    parser.add_argument(
        "--readout-time", type=float, default=None, help="Override total readout time"
    )
    parser.add_argument(
        "--eddy-options", type=str, default=None, help="Options passed to eddy"
    )
    args = parser.parse_args()
    cfg = load_config(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
