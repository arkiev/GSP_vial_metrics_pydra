# Phantom Vial Metrics Pipeline Specification

**Version:** 1.0  
**Date:** October 2024  
**Author:** Arkiev D'Souza

---

## 1. Overview

### 1.1 Purpose
Automated computation of vial-based metrics from phantom MRI scans using registration, segmentation, and quantitative analysis.

### 1.2 Scope
- **Input:** NIfTI format phantom MRI images (T1, T2, IR, multi-contrast)
- **Output:** CSV metrics files, visualization plots, registered segmentations
- **Platform:** Docker containerized, Pydra workflow management
- **Deployment:** Local execution, HPC-ready

### 1.3 Key Features
- Automated rigid-body registration to template
- Multi-contrast support with automatic detection
- Vial-based ROI analysis (mean, median, std, min, max)
- Parallel batch processing
- Session-based organization
- Workflow caching and resumability

---

## 2. Pipeline Architecture

### 2.1 High-Level Workflow

```
┌─────────────────┐
│  Input Images   │
│  (NIfTI files)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Registration   │◄─── Template Phantom
│  to Template    │◄─── Rotation Library (120 transforms)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Orientation    │
│  Validation     │◄─── QC Checks (vial intensities)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Apply Transform │
│  to Vial ROIs   │◄─── Template Vial Segmentations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Extract Metrics │
│  Per Vial       │◄─── All contrasts in session
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate Plots │
│  & Reports      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Files   │
│  (CSV + PNG)    │
└─────────────────┘
```

### 2.2 Component Breakdown

#### 2.2.1 Registration Module
- **Tool:** ANTs (antsRegistrationSyN.sh)
- **Type:** Rigid body (rotation only, no scaling/shearing)
- **Iterations:** Up to 120 (rotation library search)
- **Success Criteria:** 
  - High-intensity vials (A, O, Q) in top 5
  - Low-intensity vials (S, D, P) in bottom 5
  - Vial standard deviation < 50

#### 2.2.2 Segmentation Module
- **Input:** Template vial masks (pre-labeled)
- **Process:** Inverse transform from template → subject space
- **Output:** Subject-space vial ROIs (one per vial)

#### 2.2.3 Metrics Extraction Module
- **Tool:** MRtrix3 (mrstats)
- **Metrics:** mean, median, std, min, max
- **Per:** Each vial × each contrast × each volume
- **Output Format:** CSV matrices

#### 2.2.4 Visualization Module
- **Tool:** Python (matplotlib, seaborn)
- **Outputs:**
  - Scatter plots with error bars (mean ± std)
  - ROI overlay screenshots (MRView)
  - Parametric maps (T1, T2) if applicable

---

## 3. Data Specifications

### 3.1 Input Requirements

#### 3.1.1 Directory Structure
```
Data/
└── SessionName/          # Session identifier
    ├── scan1.nii.gz      # Primary phantom scan (T1, T2, etc.)
    ├── scan2.nii.gz      # Additional contrasts
    └── ...
```

#### 3.1.2 File Formats
- **Required:** NIfTI (.nii or .nii.gz)
- **Orientation:** Any (automatic detection and correction)
- **Dimensions:** 3D or 4D (multi-volume support)
- **Voxel Size:** Any (template resampling applied)

#### 3.1.3 Naming Conventions
- **T1-weighted:** Pattern matching `*t1*mprage*` (case-insensitive)
- **Inversion Recovery:** Pattern matching `*ir*`
- **T2/TE mapping:** Pattern matching `*TE*`
- **Flexible:** User-definable patterns

### 3.2 Output Specifications

#### 3.2.1 Directory Structure
```
output_directory/
└── SessionName/
    ├── metrics/
    │   ├── SessionName_scan1_mean_matrix.csv
    │   ├── SessionName_scan1_median_matrix.csv
    │   ├── SessionName_scan1_std_matrix.csv
    │   ├── SessionName_scan1_min_matrix.csv
    │   ├── SessionName_scan1_max_matrix.csv
    │   ├── SessionName_scan1_PLOTmeanstd.png
    │   ├── SessionName_ir_map_PLOTmeanstd_TEmapping.png
    │   └── ...
    ├── vial_segmentations/
    │   ├── VialA.nii.gz
    │   ├── VialB.nii.gz
    │   └── ...
    └── TemplatePhantom_ScannerSpace.nii.gz
```

#### 3.2.2 CSV Format
```csv
vial,scan_vol0,scan_vol1,...
VialA,1234.5,1245.2,...
VialB,987.3,992.1,...
...
```

- **Row 1:** Header (vial, contrast_vol0, contrast_vol1, ...)
- **Column 1:** Vial identifier
- **Columns 2+:** Metric values per volume

#### 3.2.3 Image Outputs
- **Format:** PNG (plots), NIfTI (segmentations)
- **Resolution:** 300 DPI (plots)
- **Color scheme:** Red ROI overlays, customizable plots

---

## 4. Execution Modes

### 4.1 Single Session Mode (pydra_basic.py)

**Purpose:** Process one phantom scan

**Usage:**
```python
python .vscode/pydra_basic.py
```

**Configuration:**
```python
input_image = "/path/to/Session/scan.nii.gz"
output_dir = "test_output"
```

**Execution Flow:**
1. Load input image path
2. Validate file exists
3. Create Pydra task
4. Execute Docker container
5. Return results

**Performance:**
- **Time:** ~5-15 minutes per session
- **Memory:** ~4 GB
- **CPU:** 8 threads (ANTs)

---

### 4.2 Batch Mode (pydra_batch.py)

**Purpose:** Process multiple sessions in parallel

**Usage:**
```bash
python .vscode/pydra_batch.py \
    /path/to/Data \
    ./batch_output \
    --pattern "*t1*mprage*.nii.gz" \
    --n-procs 4
```

**Parameters:**
- `data_dir`: Root directory containing session folders
- `output_dir`: Base output directory
- `--pattern`: Glob pattern for finding input images (default: `*t1*mprage*.nii.gz`)
- `--n-procs`: Number of parallel processes (default: 2)

**Execution Flow:**
1. Scan data directory for matching images
2. Create task list
3. Execute tasks in parallel (ProcessPoolExecutor)
4. Collect results
5. Report successes/failures

**Performance:**
- **Time:** ~5-15 min per session ÷ n_procs
- **Memory:** ~4 GB × n_procs
- **Parallelization:** Process-based (isolated containers)

---

### 4.3 Pipeline Mode (pydra_advanced.py)

**Purpose:** Advanced workflow management with caching

**Usage:**
```bash
python .vscode/pydra_advanced.py \
    /path/to/Data \
    ./pipeline_output \
    --pattern "Subject*" \
    --plugin cf \
    --n-procs 4
```

**Parameters:**
- `data_dir`: Root directory
- `output_dir`: Output directory
- `--pattern`: Session directory pattern (default: `*`)
- `--plugin`: Pydra plugin (cf=concurrent futures, serial)
- `--n-procs`: Parallel workers

**Features:**
- **Caching:** Completed sessions not reprocessed
- **Resumability:** Can restart after interruption
- **Extensible:** Class-based for adding custom steps
- **Workflow tracking:** Detailed logs and provenance

**Execution Flow:**
1. Initialize PhantomPipeline class
2. Scan for sessions matching pattern
3. Check cache for completed sessions
4. Create tasks for remaining sessions
5. Execute with Pydra workflow engine
6. Store results in cache
7. Generate summary report

**Performance:**
- **Time:** ~5-15 min per new session
- **Memory:** ~4 GB × n_procs + cache overhead
- **Cache:** Stored in `.pydra_cache/`

---

## 5. Dependencies

### 5.1 Docker Container
- **Image:** `arkiev/compute-sub-metrics:latest`
- **Base:** MRtrix3 on Debian
- **Components:**
  - MRtrix3 (latest)
  - ANTs 2.5.0
  - Python 3 (matplotlib, numpy, pandas, seaborn, scipy, pillow)
  - Xvfb (virtual display)

### 5.2 Template Data (Built into Container)
- **Template phantom:** `ImageTemplate.nii.gz`
- **Vial segmentations:** 20+ labeled vials
- **Rotation library:** 120 rigid transforms
- **Python plotting scripts:** `plot_vial_intensity.py`, `plot_maps_ir.py`, `plot_maps_TE.py`

### 5.3 Host Requirements
- **Docker:** Version 20.10+
- **Python:** 3.8+ (for Pydra)
- **Pydra:** Latest version
- **Storage:** ~2 GB per session output
- **RAM:** 4 GB minimum, 8 GB recommended
- **CPU:** Multi-core recommended for parallel processing

---

## 6. Quality Control

### 6.1 Automated QC Checks

#### 6.1.1 Registration Validation
- **Check 1:** High-intensity vials (A, O, Q) in top 5 by mean intensity
- **Check 2:** Low-intensity vials (S, D, P) in bottom 5 by mean intensity
- **Check 3:** Vial standard deviation < 50 (homogeneity check)

**Action on Failure:**
- Try next rotation from library (up to 120 attempts)
- Report iteration count in output

#### 6.1.2 File Validation
- Input file exists and readable
- NIfTI format valid
- Sufficient disk space for output

### 6.2 Manual QC Recommendations
- Review `TemplatePhantom_ScannerSpace.nii.gz` overlay
- Check vial segmentation alignment in screenshots
- Inspect metric plots for outliers
- Verify expected vial ordering (high vs low intensity)

---

## 7. Performance Specifications

### 7.1 Timing Benchmarks

| Component | Time (typical) | Time (worst case) |
|-----------|----------------|-------------------|
| Registration (single attempt) | 1-3 min | 5 min |
| Registration (with retries) | 3-10 min | 15 min |
| Metrics extraction | 30 sec - 2 min | 5 min |
| Visualization | 30 sec - 1 min | 2 min |
| **Total per session** | **5-15 min** | **30 min** |

### 7.2 Scalability

| Sessions | Mode | Time (n_procs=4) | Memory |
|----------|------|------------------|--------|
| 1 | Basic | 5-15 min | 4 GB |
| 10 | Batch | 15-40 min | 16 GB |
| 50 | Batch | 1-3 hours | 16 GB |
| 100+ | Pipeline | 2-6 hours | 16 GB + cache |

### 7.3 Resource Optimization
- **CPU:** Use `--n-procs` = number of CPU cores (max 8 per session)
- **Memory:** Limit parallel sessions to available RAM / 4 GB
- **Storage:** SSD recommended for cache and output directories

---

## 8. Error Handling

### 8.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Docker command failed` | Container not running | Check Docker Desktop running |
| `Input image not found` | Invalid path | Verify absolute path and file exists |
| `Registration failed after 120 iterations` | Poor image quality or wrong phantom | Check input image, verify phantom type |
| `No images found` | Pattern mismatch | Adjust `--pattern` to match filenames |
| `Permission denied` | Output directory permissions | Use `--user $(id -u):$(id -g)` in Docker |

### 8.2 Logging
- **stdout:** Real-time progress and results
- **stderr:** Errors and warnings
- **Docker logs:** Container execution details (captured by subprocess)
- **Pydra cache:** Task provenance and results (pipeline mode)

### 8.3 Recovery Strategies
- **Single session failure (batch mode):** Continue processing remaining sessions
- **Docker failure:** Retry mechanism not implemented (manual restart)
- **Interrupted pipeline:** Resume from cache (pipeline mode only)

---

## 9. Extensibility

### 9.1 Adding New QC Metrics

**Example:** Add SNR calculation

```python
class PhantomPipeline:
    def calculate_snr(self, session_dir):
        """Calculate signal-to-noise ratio"""
        # Read vial metrics
        mean_signal = ...
        std_noise = ...
        snr = mean_signal / std_noise
        return snr
    
    def run_with_qc(self):
        for session in self.find_sessions():
            self.run_single(session)
            snr = self.calculate_snr(session)
            print(f"SNR: {snr:.2f}")
```

### 9.2 Custom Preprocessing

**Example:** Add motion correction step

```python
class PhantomPipeline:
    def preprocess_session(self, session_dir):
        """Run motion correction before processing"""
        # Add preprocessing steps
        pass
    
    def run_batch(self, ...):
        sessions = self.find_sessions()
        for session in sessions:
            self.preprocess_session(session)  # NEW
            self.run_single(session)
```

### 9.3 Alternative Docker Images

**Example:** Use custom Docker image

```python
class PhantomPipeline:
    def __init__(self, data_dir, output_dir, docker_image=None):
        self.docker_image = docker_image or "arkiev/compute-sub-metrics:latest"
    
    def create_task(self, ...):
        cmd = ["docker", "run", self.docker_image, ...]  # Uses custom image
```

### 9.4 Output Format Changes

**Example:** Add JSON output

```python
class PhantomPipeline:
    def export_to_json(self, session_name):
        """Convert CSV metrics to JSON"""
        import json
        import pandas as pd
        
        csv_file = self.output_dir / session_name / "metrics" / "mean_matrix.csv"
        df = pd.read_csv(csv_file)
        json_file = csv_file.with_suffix('.json')
        df.to_json(json_file, orient='records')
```

---

## 10. Validation & Testing

### 10.1 Unit Tests (Future Work)
- Test rotation library loading
- Test registration validation logic
- Test CSV writing/reading
- Test path handling (cross-platform)

### 10.2 Integration Tests (Future Work)
- End-to-end processing of test phantom
- Batch processing of multiple test cases
- Verify output file structure
- Validate metric accuracy against ground truth

### 10.3 Performance Tests (Future Work)
- Benchmark processing time per session
- Memory profiling
- Parallel efficiency testing
- Cache performance evaluation

---

## 11. Deployment

### 11.1 Local Installation

**Prerequisites:**
1. Install Docker Desktop
2. Install Python 3.8+
3. Create virtual environment
4. Install Pydra: `pip install pydra`
5. Pull Docker image: `docker pull arkiev/compute-sub-metrics:latest`

**Setup:**
```bash
git clone <repository>
cd GSP_vial_metrics
python -m venv venv  # or use pyenv
source venv/bin/activate
pip install pydra
python .vscode/test_setup.py  # Verify installation
```

### 11.2 HPC Deployment (Future Work)

**Considerations:**
- Replace Docker with Singularity (HPC-compatible)
- Use Pydra SLURM plugin for job submission
- Shared filesystem for cache and outputs
- Module system for dependencies

**Example SLURM submission:**
```bash
sbatch --array=1-100 --cpus-per-task=8 --mem=8G \
    pydra_batch_slurm.sh /data/phantoms /results
```

### 11.3 Cloud Deployment (Future Work)

**Options:**
- AWS Batch with Docker containers
- Google Cloud Run
- Azure Container Instances

**Considerations:**
- Data transfer costs
- Storage (S3, Cloud Storage)
- Compute costs
- Egress bandwidth

---

## 12. Maintenance & Support

### 12.1 Version Control
- **Repository:** GitHub (recommended)
- **Branching:** main (stable), dev (development)
- **Releases:** Semantic versioning (v1.0.0, v1.1.0, ...)
- **Docker tags:** Match release versions

### 12.2 Update Procedures

**Docker Image Updates:**
1. Rebuild container: `docker build -t compute-sub-metrics:v1.1 .`
2. Tag: `docker tag compute-sub-metrics:v1.1 arkiev/compute-sub-metrics:v1.1`
3. Push: `docker push arkiev/compute-sub-metrics:v1.1`
4. Update scripts to reference new version

**Python Code Updates:**
1. Update scripts in `.vscode/`
2. Test with `test_setup.py`
3. Update version in documentation
4. Commit and push to repository

### 12.3 Documentation
- **README.md:** Quick start guide
- **SETUP_MAC.md:** Platform-specific setup
- **PIPELINE_SPEC.md:** This document
- **API docs:** (Future work - Sphinx or MkDocs)

---

## 13. Future Enhancements

### 13.1 Planned Features
- [ ] Web interface for job submission and monitoring
- [ ] Automated report generation (PDF)
- [ ] Database backend for metrics storage
- [ ] Longitudinal analysis tools
- [ ] Group statistic
