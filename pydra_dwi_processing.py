#!/usr/bin/env python3
"""
DWI processing and tensor metrics pipeline (Pydra ~0.22)

Usage:
    python dwi_processing.py --config config.yaml
    python dwi_processing.py --scans-dir /path/to/scans [options]

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
import sys
from pathlib import Path

import pydra
import yaml


# =============================================================================
# Directory scanning helpers (pure Python, run before Pydra graph)
# =============================================================================


def strip_series_number(name: str) -> str:
    """Strip leading series number, e.g. '87-ep2d_diff_...' -> 'ep2d_diff_...'"""
    return re.sub(r"^\d+-", "", name)


def common_prefix_len(a: str, b: str) -> int:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def find_best_pe_dir(dwi_name: str, pe_dirs: list) -> str | None:
    """
    Find the PE directory best matching a given DWI name.
    Uses longest common prefix (after stripping series numbers).
    Falls back to first available shared PE dir if no strong match (>10 chars).
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

    # Fall back to first shared PE dir
    return pe_dirs[0] if pe_dirs else None


def scan_directory(scans_dir: str) -> dict:
    """
    Scan a scans directory and classify subdirectories into:
      - t1_dir: single T1 directory
      - dwi_dirs: list of main DWI directories
      - fwd_pe_dirs: list of forward PE directories
      - rpe_dirs: list of reverse PE directories
    """
    scans_path = Path(scans_dir)
    t1_dir = None
    dwi_dirs = []
    fwd_pe_dirs = []
    rpe_dirs = []

    for d in sorted(scans_path.iterdir()):
        if not d.is_dir():
            continue
        name = d.name

        if re.search(r"t1", name, re.IGNORECASE):
            t1_dir = str(d)
            continue

        if re.search(r"_(ADC|FA)$", name, re.IGNORECASE):
            continue

        if re.search(r"_(A_P|AP|L_R|LR|S_I|SI)$", name, re.IGNORECASE):
            fwd_pe_dirs.append(str(d))
            continue

        if re.search(r"_(P_A|PA|R_L|RL|I_S|IS)$", name, re.IGNORECASE):
            rpe_dirs.append(str(d))
            continue

        dwi_dirs.append(str(d))

    if t1_dir is None:
        raise ValueError(f"Could not identify T1 directory in {scans_dir}")
    if not dwi_dirs:
        raise ValueError(f"Could not identify any DWI directories in {scans_dir}")

    return {
        "t1_dir": t1_dir,
        "dwi_dirs": dwi_dirs,
        "fwd_pe_dirs": fwd_pe_dirs,
        "rpe_dirs": rpe_dirs,
    }


def detect_pe_direction(folder_name: str) -> tuple[str, str]:
    """Return (pe_dir, rpe_dir) from a DWI folder name."""
    pairs = [
        (r"_(A_P|AP)(_|$)", "AP", "PA"),
        (r"_(P_A|PA)(_|$)", "PA", "AP"),
        (r"_(L_R|LR)(_|$)", "LR", "RL"),
        (r"_(R_L|RL)(_|$)", "RL", "LR"),
        (r"_(S_I|SI)(_|$)", "SI", "IS"),
        (r"_(I_S|IS)(_|$)", "IS", "SI"),
    ]
    for pattern, pe, rpe in pairs:
        if re.search(pattern, folder_name, re.IGNORECASE):
            return pe, rpe
    raise ValueError(
        f"Could not determine PE direction from folder name: {folder_name}\n"
        f"Expected one of: AP, PA, LR, RL, SI, IS (or underscore variants)"
    )


def get_nvols(nii_path: str) -> int:
    """Return number of volumes using mrinfo -ndim."""
    result = subprocess.run(
        ["mrinfo", nii_path, "-ndim"], capture_output=True, text=True, check=True
    )
    ndim = int(result.stdout.strip())
    if ndim == 3:
        return 1
    result2 = subprocess.run(
        ["mrinfo", nii_path, "-size"], capture_output=True, text=True, check=True
    )
    return int(result2.stdout.strip().split()[-1])


def determine_preproc_mode(
    rpe_nii: str | None, rpe_nvols: int, dwi_nvols: int, fwd_pe_nii: str | None
) -> str:
    if rpe_nii is None:
        return "rpe_none"
    if rpe_nvols == dwi_nvols:
        return "rpe_all"
    if rpe_nvols > 1 and fwd_pe_nii is not None:
        return "rpe_split"
    return "rpe_pair"


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
    print(f"Warning: TotalReadoutTime not found in {json_path} -- using {fallback}")
    return fallback


def run_cmd(cmd: list, cwd: str = None):
    """Run a shell command, raising on failure."""
    print(f"  >> {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


# =============================================================================
# Pydra tasks
# =============================================================================


@pydra.mark.task
def convert_dicoms(dicom_dir: str, out_dir: str) -> dict:
    """
    Convert a DICOM directory to NIfTI using dcm2niix.
    Returns paths to the first NIfTI and its sidecar files.
    """
    os.makedirs(out_dir, exist_ok=True)
    run_cmd(["dcm2niix", "-o", out_dir, "-f", "%p", "-z", "y", dicom_dir])

    niis = sorted(Path(out_dir).glob("*.nii.gz"))
    if not niis:
        raise FileNotFoundError(f"No NIfTI files produced in {out_dir}")

    nii = str(niis[0])
    base = nii.replace(".nii.gz", "")
    return {
        "nii": nii,
        "json": base + ".json" if Path(base + ".json").exists() else "",
        "bvec": base + ".bvec" if Path(base + ".bvec").exists() else "",
        "bval": base + ".bval" if Path(base + ".bval").exists() else "",
    }


@pydra.mark.task
def run_gradcheck(nii: str, bvec: str, bval: str, out_dir: str, prefix: str) -> dict:
    """Run dwigradcheck and return corrected bvec/bval paths."""
    out_bvec = str(Path(out_dir) / f"{prefix}_corrected.bvec")
    out_bval = str(Path(out_dir) / f"{prefix}_corrected.bval")
    run_cmd(
        [
            "dwigradcheck",
            nii,
            "-export_grad_fsl",
            out_bvec,
            out_bval,
            "-fslgrad",
            bvec,
            bval,
        ]
    )
    return {"bvec": out_bvec, "bval": out_bval}


@pydra.mark.task
def convert_to_mif(
    nii: str, json_file: str, bvec: str, bval: str, out_path: str
) -> str:
    """Convert NIfTI to MIF using mrconvert with JSON and FSL gradients."""
    cmd = [
        "mrconvert",
        nii,
        out_path,
        "-json_import",
        json_file,
        "-fslgrad",
        bvec,
        bval,
    ]
    run_cmd(cmd)

    # Verify PE table embedded
    result = subprocess.run(
        ["mrinfo", out_path, "-petable"], capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"Warning: PE table not found in MIF header: {out_path}")
        print(
            f"         Check {json_file} contains PhaseEncodingDirection and TotalReadoutTime"
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
    Build the se_epi pair for dwifslpreproc (rpe_pair / rpe_split).
    Returns path to bzero_pair.mif.gz, or empty string if not needed.
    """
    if preproc_mode not in ("rpe_pair", "rpe_split"):
        return ""

    # Convert RPE image
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

    # Forward PE: use dedicated image or compute mean b0 from DWI
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
        print("No forward PE image -- computing mean b0 from DWI...")
        mean_b0 = str(Path(tmp_dir) / f"mean_bzero_{pe_dir}.mif.gz")
        # dwiextract | mrmath pipeline
        extract = subprocess.Popen(
            ["dwiextract", dwi_mif, "-", "-bzero"], stdout=subprocess.PIPE
        )
        with open(mean_b0, "wb") as _:
            pass  # placeholder — mrmath reads stdin
        result = subprocess.run(
            ["mrmath", "-", "mean", mean_b0, "-axis", "3"],
            stdin=extract.stdout,
            check=True,
        )
        extract.stdout.close()
        extract.wait()
        fwd_mif = mean_b0

    # Concatenate: forward PE first, reverse PE last
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
    rpe_nii: str,
    rpe_json: str,
    rpe_bvec: str,
    rpe_bval: str,
    rpe_dir: str,
    tmp_dir: str,
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
        cmd += ["-se_epi", se_epi, "-readout_time", str(readout_time)]
    elif preproc_mode == "rpe_all":
        # Convert RPE for rpe_all
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
    """N4 bias field correction for T1."""
    run_cmd(["N4BiasFieldCorrection", "-i", t1_nii, "-o", out_nii])
    return out_nii


@pydra.mark.task
def extract_mean_b0(dwi_biascorr_mif: str, out_nii: str) -> str:
    """Extract mean b0 from bias-corrected DWI for registration."""
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
def register_b0_to_t1(
    b0_nii: str, t1_nii: str, out_b0_in_t1: str, out_mat: str
) -> dict:
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
    return {"b02t1_mat": out_mat}


@pydra.mark.task
def invert_and_apply_transform(
    b02t1_mat: str, t1_nii: str, b0_nii: str, tmp_dir: str
) -> dict:
    """
    Invert the b0->T1 transform and apply it to resample T1 into DWI space.
    Also converts the inverse transform to MRtrix format.
    """
    t12b0_mat = str(Path(tmp_dir) / "T12b0.mat")
    t1_in_dwi = str(Path(tmp_dir) / "T1_n4_in_DWI_space.nii.gz")
    mrtrix_txt = str(Path(tmp_dir) / "struct2diff_mrtrix.txt")

    # Invert transform
    run_cmd(["convert_xfm", "-omat", t12b0_mat, "-inverse", b02t1_mat])

    # Apply inverse: resample T1 into DWI space
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

    # Convert to MRtrix format
    run_cmd(["transformconvert", t12b0_mat, t1_nii, b0_nii, "flirt_import", mrtrix_txt])

    return {"t1_in_dwi": t1_in_dwi, "mrtrix_xfm": mrtrix_txt}


@pydra.mark.task
def compute_tensor_metrics(dwi_biascorr_mif: str, tmp_dir: str) -> dict:
    """Run dwi2tensor and tensor2metric, returning NIfTI ADC and FA paths."""
    tensor_mif = str(Path(tmp_dir) / "tensor.mif.gz")
    adc_mif = str(Path(tmp_dir) / "ADC.mif.gz")
    fa_mif = str(Path(tmp_dir) / "FA.mif.gz")
    adc_nii = str(Path(tmp_dir) / "ADC.nii.gz")
    fa_nii = str(Path(tmp_dir) / "FA.nii.gz")

    run_cmd(["dwi2tensor", dwi_biascorr_mif, tensor_mif])
    run_cmd(["tensor2metric", "-adc", adc_mif, "-fa", fa_mif, tensor_mif])
    run_cmd(["mrconvert", adc_mif, adc_nii])
    run_cmd(["mrconvert", fa_mif, fa_nii])

    return {"adc": adc_nii, "fa": fa_nii}


@pydra.mark.task
def copy_final_outputs(
    dwi_biascorr_mif: str,
    t1_in_dwi: str,
    adc_nii: str,
    fa_nii: str,
    out_dir: str,
    dwi_preproc_name: str,
) -> dict:
    """Copy the four key outputs from tmp/ to the parent output directory."""
    os.makedirs(out_dir, exist_ok=True)

    outputs = {
        dwi_biascorr_mif: str(Path(out_dir) / dwi_preproc_name),
        t1_in_dwi: str(Path(out_dir) / "T1_n4_in_DWI_space.nii.gz"),
        adc_nii: str(Path(out_dir) / "ADC.nii.gz"),
        fa_nii: str(Path(out_dir) / "FA.nii.gz"),
    }
    for src, dst in outputs.items():
        shutil.copy2(src, dst)
        print(f"  Copied: {Path(dst).name}")

    return {
        "dwi_biascorr": outputs[dwi_biascorr_mif],
        "t1_in_dwi": outputs[t1_in_dwi],
        "adc": outputs[adc_nii],
        "fa": outputs[fa_nii],
    }


# =============================================================================
# Per-DWI pipeline (Pydra Workflow)
# =============================================================================


def build_dwi_workflow(
    dwi_dir: str,
    t1_dir: str,
    fwd_pe_dirs: list,
    rpe_dirs: list,
    parent_out_dir: str,
    cfg: dict,
) -> pydra.Workflow:
    """
    Build a Pydra workflow for a single DWI series.
    T1 bias correction runs in parallel with DWI preprocessing.
    """
    dwi_name = Path(dwi_dir).name
    pe_dir, rpe_dir = detect_pe_direction(dwi_name)
    fwd_pe_dir = find_best_pe_dir(dwi_name, fwd_pe_dirs)
    rpe_dir_path = find_best_pe_dir(dwi_name, rpe_dirs)

    out_dir = str(Path(parent_out_dir) / dwi_name)
    tmp_dir = str(Path(out_dir) / "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    do_denoise = cfg.get("denoise_degibbs", False)
    do_gradcheck = cfg.get("gradcheck", False)
    eddy_options = cfg.get("eddy_options", " --slm=linear")
    readout_time_override = cfg.get("readout_time", None)

    dwi_preproc_name = (
        "DWI_denoise_gibbs_preproc_biascorr.mif.gz"
        if do_denoise
        else "DWI_preproc_biascorr.mif.gz"
    )

    wf = pydra.Workflow(name=f"dwi_{dwi_name}", input_spec=["x"])
    wf.inputs.x = 1  # dummy input to trigger execution

    # ------------------------------------------------------------------
    # Step 1a: Convert DWI DICOMs
    # ------------------------------------------------------------------
    wf.add(
        convert_dicoms(
            name="convert_dwi",
            dicom_dir=dwi_dir,
            out_dir=str(Path(tmp_dir) / "dwi_nii"),
        )
    )

    # ------------------------------------------------------------------
    # Step 1b: Convert T1 DICOMs (runs in parallel with DWI chain)
    # ------------------------------------------------------------------
    wf.add(
        convert_dicoms(
            name="convert_t1",
            dicom_dir=t1_dir,
            out_dir=str(Path(tmp_dir) / "t1_nii"),
        )
    )

    # ------------------------------------------------------------------
    # Step 1c: Convert forward PE DICOMs (if present)
    # ------------------------------------------------------------------
    if fwd_pe_dir:
        wf.add(
            convert_dicoms(
                name="convert_fwd_pe",
                dicom_dir=fwd_pe_dir,
                out_dir=str(Path(tmp_dir) / "fwd_pe_nii"),
            )
        )

    # ------------------------------------------------------------------
    # Step 1d: Convert reverse PE DICOMs (if present)
    # ------------------------------------------------------------------
    if rpe_dir_path:
        wf.add(
            convert_dicoms(
                name="convert_rpe",
                dicom_dir=rpe_dir_path,
                out_dir=str(Path(tmp_dir) / "rpe_nii"),
            )
        )

    # ------------------------------------------------------------------
    # Step 2: T1 N4 bias correction (parallel with DWI chain)
    # ------------------------------------------------------------------
    wf.add(
        run_n4(
            name="n4_t1",
            t1_nii=wf.convert_t1.lzout.nii,
            out_nii=str(Path(tmp_dir) / "T1_n4.nii.gz"),
        )
    )

    # ------------------------------------------------------------------
    # Step 3: Optional gradcheck on DWI (and PE images)
    # ------------------------------------------------------------------
    if do_gradcheck:
        wf.add(
            run_gradcheck(
                name="gradcheck_dwi",
                nii=wf.convert_dwi.lzout.nii,
                bvec=wf.convert_dwi.lzout.bvec,
                bval=wf.convert_dwi.lzout.bval,
                out_dir=tmp_dir,
                prefix="dwi",
            )
        )
        dwi_bvec = wf.gradcheck_dwi.lzout.bvec
        dwi_bval = wf.gradcheck_dwi.lzout.bval

        if fwd_pe_dir:
            wf.add(
                run_gradcheck(
                    name="gradcheck_fwd_pe",
                    nii=wf.convert_fwd_pe.lzout.nii,
                    bvec=wf.convert_fwd_pe.lzout.bvec,
                    bval=wf.convert_fwd_pe.lzout.bval,
                    out_dir=tmp_dir,
                    prefix="fwd_pe",
                )
            )
            fwd_bvec = wf.gradcheck_fwd_pe.lzout.bvec
            fwd_bval = wf.gradcheck_fwd_pe.lzout.bval
        else:
            fwd_bvec = fwd_bval = ""

        if rpe_dir_path:
            wf.add(
                run_gradcheck(
                    name="gradcheck_rpe",
                    nii=wf.convert_rpe.lzout.nii,
                    bvec=wf.convert_rpe.lzout.bvec,
                    bval=wf.convert_rpe.lzout.bval,
                    out_dir=tmp_dir,
                    prefix="rpe",
                )
            )
            rpe_bvec = wf.gradcheck_rpe.lzout.bvec
            rpe_bval = wf.gradcheck_rpe.lzout.bval
        else:
            rpe_bvec = rpe_bval = ""
    else:
        dwi_bvec = wf.convert_dwi.lzout.bvec
        dwi_bval = wf.convert_dwi.lzout.bval
        fwd_bvec = wf.convert_fwd_pe.lzout.bvec if fwd_pe_dir else ""
        fwd_bval = wf.convert_fwd_pe.lzout.bval if fwd_pe_dir else ""
        rpe_bvec = wf.convert_rpe.lzout.bvec if rpe_dir_path else ""
        rpe_bval = wf.convert_rpe.lzout.bval if rpe_dir_path else ""

    # ------------------------------------------------------------------
    # Step 4: Convert DWI to MIF
    # ------------------------------------------------------------------
    wf.add(
        convert_to_mif(
            name="dwi_to_mif",
            nii=wf.convert_dwi.lzout.nii,
            json_file=wf.convert_dwi.lzout.json,
            bvec=dwi_bvec,
            bval=dwi_bval,
            out_path=str(Path(tmp_dir) / f"DWI_raw_{pe_dir}.mif.gz"),
        )
    )

    # ------------------------------------------------------------------
    # Step 5: Optional denoise + Gibbs
    # ------------------------------------------------------------------
    if do_denoise:
        wf.add(
            run_dwidenoise(
                name="denoise",
                in_mif=wf.dwi_to_mif.lzout.out,
                out_mif=str(Path(tmp_dir) / f"DWI_raw_{pe_dir}_denoise.mif.gz"),
            )
        )
        wf.add(
            run_mrdegibbs(
                name="degibbs",
                in_mif=wf.denoise.lzout.out,
                out_mif=str(Path(tmp_dir) / f"DWI_raw_{pe_dir}_denoise_gibbs.mif.gz"),
            )
        )
        dwi_for_preproc = wf.degibbs.lzout.out
    else:
        dwi_for_preproc = wf.dwi_to_mif.lzout.out

    # ------------------------------------------------------------------
    # Determine preproc mode (needs concrete NIfTI paths — resolved now)
    # ------------------------------------------------------------------
    rpe_nii_path = None
    fwd_nii_path = None
    rpe_json_path = ""
    fwd_json_path = ""

    if rpe_dir_path:
        rpe_nii_path = str(next(Path(tmp_dir, "rpe_nii").glob("*.nii.gz"), ""))
    if fwd_pe_dir:
        fwd_nii_path = str(next(Path(tmp_dir, "fwd_pe_nii").glob("*.nii.gz"), ""))

    # Volume counts need the actual files; resolved after dcm2niix.
    # We use a lazy approach: determine mode at workflow execution via
    # a dedicated task.
    @pydra.mark.task
    def resolve_preproc_mode(rpe_nii: str, dwi_nii: str, fwd_nii: str) -> str:
        if not rpe_nii or not Path(rpe_nii).exists():
            return "rpe_none"
        rpe_nvols = get_nvols(rpe_nii)
        dwi_nvols = get_nvols(dwi_nii)
        if rpe_nvols == dwi_nvols:
            return "rpe_all"
        if rpe_nvols > 1 and fwd_nii and Path(fwd_nii).exists():
            return "rpe_split"
        return "rpe_pair"

    wf.add(
        resolve_preproc_mode(
            name="preproc_mode",
            rpe_nii=wf.convert_rpe.lzout.nii if rpe_dir_path else "",
            dwi_nii=wf.convert_dwi.lzout.nii,
            fwd_nii=wf.convert_fwd_pe.lzout.nii if fwd_pe_dir else "",
        )
    )

    # ------------------------------------------------------------------
    # Resolve readout time (lazy, from JSON after conversion)
    # ------------------------------------------------------------------
    @pydra.mark.task
    def resolve_readout_time(json_path: str, override: float) -> float:
        if override is not None:
            return override
        return get_readout_time(json_path)

    wf.add(
        resolve_readout_time(
            name="readout_time",
            json_path=wf.convert_dwi.lzout.json,
            override=readout_time_override,
        )
    )

    # ------------------------------------------------------------------
    # Step 6: Build se_epi pair
    # ------------------------------------------------------------------
    wf.add(
        build_se_epi(
            name="se_epi",
            dwi_mif=dwi_for_preproc,
            rpe_nii=wf.convert_rpe.lzout.nii if rpe_dir_path else "",
            rpe_json=wf.convert_rpe.lzout.json if rpe_dir_path else "",
            rpe_bvec=rpe_bvec,
            rpe_bval=rpe_bval,
            fwd_pe_nii=wf.convert_fwd_pe.lzout.nii if fwd_pe_dir else "",
            fwd_pe_json=wf.convert_fwd_pe.lzout.json if fwd_pe_dir else "",
            fwd_pe_bvec=fwd_bvec,
            fwd_pe_bval=fwd_bval,
            pe_dir=pe_dir,
            rpe_dir=rpe_dir,
            preproc_mode=wf.preproc_mode.lzout.out,
            tmp_dir=tmp_dir,
        )
    )

    # ------------------------------------------------------------------
    # Step 7: dwifslpreproc
    # ------------------------------------------------------------------
    wf.add(
        run_dwifslpreproc(
            name="fslpreproc",
            dwi_mif=dwi_for_preproc,
            out_mif=str(Path(tmp_dir) / "DWI_preproc.mif.gz"),
            pe_dir=pe_dir,
            preproc_mode=wf.preproc_mode.lzout.out,
            se_epi=wf.se_epi.lzout.out,
            readout_time=wf.readout_time.lzout.out,
            eddy_options=eddy_options,
            rpe_nii=wf.convert_rpe.lzout.nii if rpe_dir_path else "",
            rpe_json=wf.convert_rpe.lzout.json if rpe_dir_path else "",
            rpe_bvec=rpe_bvec,
            rpe_bval=rpe_bval,
            rpe_dir=rpe_dir,
            tmp_dir=tmp_dir,
        )
    )

    # ------------------------------------------------------------------
    # Step 8: Mask + DWI bias correction
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
    # Step 9: Extract mean b0 (downstream of DWI biascorr)
    # ------------------------------------------------------------------
    wf.add(
        extract_mean_b0(
            name="mean_b0",
            dwi_biascorr_mif=wf.dwi_biascorr.lzout.out,
            out_nii=str(Path(tmp_dir) / "bzero_f.nii.gz"),
        )
    )

    # ------------------------------------------------------------------
    # Step 10: Registration (downstream of both mean_b0 AND n4_t1)
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
    # Step 11: Tensor metrics
    # ------------------------------------------------------------------
    wf.add(
        compute_tensor_metrics(
            name="tensor",
            dwi_biascorr_mif=wf.dwi_biascorr.lzout.out,
            tmp_dir=tmp_dir,
        )
    )

    # ------------------------------------------------------------------
    # Step 12: Copy final outputs
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
# Top-level pipeline: one workflow per DWI, submitted in parallel
# =============================================================================


def run_pipeline(cfg: dict):
    scans_dir = cfg["scans_dir"]
    parent_out_dir = cfg.get("output_dir", str(Path(scans_dir).name))
    os.makedirs(parent_out_dir, exist_ok=True)

    print(f"Scans directory:  {scans_dir}")
    print(f"Denoise/Degibbs:  {cfg.get('denoise_degibbs', False)}")
    print(f"Gradcheck:        {cfg.get('gradcheck', False)}")
    print(f"Output:           {Path(parent_out_dir).resolve()}")

    dirs = scan_directory(scans_dir)
    print(f"T1: {dirs['t1_dir']}")
    print(
        f"Found {len(dirs['dwi_dirs'])} DWI series, "
        f"{len(dirs['fwd_pe_dirs'])} forward PE, "
        f"{len(dirs['rpe_dirs'])} reverse PE"
    )

    # Build one sub-workflow per DWI series
    sub_workflows = []
    for dwi_dir in dirs["dwi_dirs"]:
        print(f"\nBuilding workflow for: {Path(dwi_dir).name}")
        wf = build_dwi_workflow(
            dwi_dir=dwi_dir,
            t1_dir=dirs["t1_dir"],
            fwd_pe_dirs=dirs["fwd_pe_dirs"],
            rpe_dirs=dirs["rpe_dirs"],
            parent_out_dir=parent_out_dir,
            cfg=cfg,
        )
        sub_workflows.append(wf)

    # Wrap all DWI workflows in a top-level workflow
    # Each DWI sub-workflow is independent — Pydra can execute them in parallel
    top = pydra.Workflow(name="dwi_pipeline", input_spec=["x"])
    top.inputs.x = 1

    for wf in sub_workflows:
        top.add(wf)

    # Collect outputs from all sub-workflows
    top.set_output(
        [
            (f"out_{wf.name}", getattr(top, wf.name).lzout.outputs)
            for wf in sub_workflows
        ]
    )

    with pydra.Submitter(plugin="cf") as sub:
        sub(top)

    results = top.result()
    print("\n=== All done ===")
    print(f"Outputs in: {Path(parent_out_dir).resolve()}")
    return results


# =============================================================================
# CLI
# =============================================================================


def load_config(args) -> dict:
    """Merge YAML config (if provided) with CLI arguments. CLI takes priority."""
    cfg = {}

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides
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
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument(
        "--scans-dir", type=str, help="Path to scans directory (overrides config)"
    )
    parser.add_argument(
        "--denoise-degibbs",
        action="store_true",
        default=None,
        help="Apply dwidenoise and mrdegibbs (overrides config)",
    )
    parser.add_argument(
        "--gradcheck",
        action="store_true",
        default=None,
        help="Apply dwigradcheck (overrides config)",
    )
    parser.add_argument(
        "--readout-time",
        type=float,
        default=None,
        help="Override total readout time (overrides JSON + config)",
    )
    parser.add_argument(
        "--eddy-options",
        type=str,
        default=None,
        help="Options passed to eddy (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to parent output directory (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config(args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
