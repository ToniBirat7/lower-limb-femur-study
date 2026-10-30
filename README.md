# Quantitative 3D Lower-Limb CT Analysis & Surgical Planning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.6.1-008080)](https://monai.io/)
[![nnU--Net](https://img.shields.io/badge/nnU--Net-v2.8.1-blue)](https://github.com/MIC-DKFZ/nnUNet)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end clinical engineering study on **3D bone segmentation, anatomical landmark detection, orthopaedic resection planning, and 2D-versus-3D radiographic projection analysis** on 101 whole-femur CT scans from the HFValid cohort.

Developed for research translation aligned with the **Nepal Applied Mathematics and Informatics Institute for Research (NAAMII)** Multimodal Medical Data Analysis for Precision Medicine (MAPMED) platform.

---

## 1. Study Highlights & Clinical Objectives

1. **Benchmark 4 Segmentation Methods on Identical Clinical Test Split:**
   - Classical intensity windowing ($[200, 1800]$ HU) + 3D connected components + morphology.
   - Custom 3D U-Net with residual units and mixed-precision sliding-window inference (MONAI).
   - Self-configuring deep network (`nnU-Net v2 3d_fullres`).
   - Out-of-the-box anatomical foundation model (`TotalSegmentator CT`).
2. **Clinical Metric Rigor:**
   - Evaluated using Dice Similarity Coefficient (DSC), 95th Percentile Hausdorff Distance (HD95 in mm), and Average Surface Distance (ASD in mm).
   - Paired non-parametric Wilcoxon signed-rank significance testing and bootstrap 95% confidence intervals.
3. **Orthopaedic Planning Measurements (Stage 1f):**
   - **Femoral Head Diameter:** Least-squares sphere fitting for hip arthroplasty implant sizing.
   - **Collo-Diaphyseal / Neck-Shaft Angle (CCD):** Measured between the 3D femoral neck vector and the anatomical shaft axis.
   - **Femoral Anteversion:** Rotational torsion angle in the plane orthogonal to the diaphysis.
   - **Mechanical-Anatomical Angle (AMA):** Used clinically to set the distal femoral cut angle in total knee arthroplasty (resection planning).
   - Predicted meshes validated against ground-truth meshes with Bland-Altman agreement analysis.
4. **Hypothesis-Driven 2D vs. 3D Radiographic Distortion Experiment (Stage 1g):**
   - Digitally forward-projected CT volumes into simulated front-view radiographs across rotational angles ($-20^\circ$ to $+20^\circ$).
   - Quantifies the clinical risk of rotational malposition on 2D radiographic deformity measurement.

---

## 2. Dataset Overview: HFValid Cohort v3

- **Cohort:** 101 female patients ($72.4 \pm 7.1$ years, range 55 to 88) scanned at IRCCS Istituto Ortopedico Rizzoli (IOR), Italy.
- **Scanners:** GE Medical Systems (BrightSpeed, HiSpeed CT/i, LightSpeed16).
- **In-Plane Spacing:** $0.764 \pm 0.061$ mm ($0.586$ to $0.977$ mm).
- **Slice Thickness:** $1.0$ to $2.0$ mm ($1.83 \pm 0.27$ mm).
- **Ground Truth:** Calibrated CT scans (`.nrrd`), expert surface meshes (`.stl`), and four clinical landmarks (`femurHead`, `headDir`, `greatTroch`, `kneeCenter` in `_palp.inp`).

Detailed dataset documentation: [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

---

## 3. Repository Structure

```text
├── src/
│   ├── __init__.py             # Python package indicator
│   ├── audit.py                # 101-scan dataset audit and data card generator
│   ├── voxelize.py             # Parallel STL-to-mask voxelization (VTK / SimpleITK)
│   ├── split_data.py           # Deterministic patient-level train/val/test split
│   ├── metrics.py              # Unit-tested Dice, HD95 (mm), and ASD (mm)
│   ├── geometry.py             # Least-squares sphere fit, line fit, 3D angle projection
│   ├── classical_baseline.py   # HU windowing + 3D connected component baseline
│   ├── convert_nnunet.py       # nnU-Net v2 raw dataset formatter (Dataset001_Femur)
│   ├── planning.py             # Implant sizing & resection angle calculations (1f)
│   └── xray_projection.py      # 2D projection & rotational error experiment (1g)
├── tests/
│   ├── test_geometry.py        # Computational geometry unit tests (pytest)
│   └── test_metrics.py         # Segmentation metric unit tests (pytest)
├── docs/
│   ├── DATA_CARD.md            # Clinical dataset profile & imaging characteristics
│   └── SETUP.md                # Environment activation and reproduction guide
├── pyproject.toml              # Build config, dependencies & test specifications
├── splits.json                 # Patient-level split (71 train / 10 val / 20 test)
├── hfvalid_audit.csv           # Full audit metrics for all 101 subjects
└── README.md                   # Project overview & clinical results
```

---

## 4. Quickstart

### Environment Setup
```bash
# Clone and activate Python 3.12 environment
git clone <repo-url>
cd lower-limb-femur-study
source .venv/bin/activate

# Run metrics and geometry unit tests via pytest
pytest
```

### Reproducing Baseline Results
```bash
# 1. Inspect dataset
python src/audit.py

# 2. Run classical segmentation baseline on validation split
python src/classical_baseline.py val
```

---

## 5. Methodological & Regulatory Standards
- **CLAIM Compliance:** Structured according to the 2024 RSNA Checklist for Artificial Intelligence in Medical Imaging.
- **Zero Leakage:** Strictly disjoint patient-level partitioning; test set evaluated once.
- **Traceability:** Software lifecycle standards aligned with IEC 62304 and FDA 510(k) verification principles.
