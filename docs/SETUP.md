# Setup & Reproduction Guide

## 1. Hardware & System Architecture
- **Operating System:** Linux (Arch Linux x86_64)
- **GPU:** NVIDIA GeForce RTX 3060 Laptop GPU (6,144 MiB VRAM, Compute Capability 8.6)
- **CUDA Runtime:** 12.4
- **Storage Configuration:**
  - **NVMe SSD Partition (`EA1AD7931AD75AE1`):** Repository code, unit tests, fast Python `.venv`.
  - **External HDD (`TOSHIBA_EXT`):** Raw CT volumes, voxelized masks, nnU-Net preprocessed data, model weights, 3D Slicer, and paper PDFs.
  - **Root Partition (`/`):** Preserved with zero project files or cache writes.

---

## 2. Environment Activation
The project uses a high-performance Python 3.12 environment managed via `uv`:

```bash
cd "/run/media/tonibirat/EA1AD7931AD75AE1/Projects/lower-limb-femur-study"
source .venv/bin/activate
```

### Verified Core Dependencies
- `torch 2.6.0+cu124` / `torchvision 0.21.0+cu124`
- `SimpleITK 2.5.6`
- `monai 1.6.1`
- `pyvista 0.49.2` / `vtk 9.7.1`
- `totalsegmentator 2.18.0`
- `nnunetv2 2.8.1`
- `scipy 1.18.1` / `scikit-image 0.26.0`

---

## 3. Directory Layout & Data Pointers
The code accesses data on the external drive via explicit paths:

- Raw HFValid Dataset: `/run/media/tonibirat/TOSHIBA_EXT/naamii-data/HFValid/Subjects/`
- Ground-Truth Binary Masks: `/run/media/tonibirat/TOSHIBA_EXT/naamii-data/HFValid/masks/`
- TotalSegmentator Weights: Symlinked to `/run/media/tonibirat/TOSHIBA_EXT/weights/totalsegmentator`
- nnU-Net Environment Variables:
  ```bash
  export nnUNet_raw="/run/media/tonibirat/TOSHIBA_EXT/naamii-data/nnunet/nnUNet_raw"
  export nnUNet_preprocessed="/run/media/tonibirat/TOSHIBA_EXT/naamii-data/nnunet/nnUNet_preprocessed"
  export nnUNet_results="/run/media/tonibirat/TOSHIBA_EXT/naamii-data/nnunet/nnUNet_results"
  ```

---

## 4. Execution Pipeline

### A. Run Data Audit
```bash
python src/audit.py
```
Outputs: `hfvalid_audit.csv` with dimensions, spacing, HU statistics, and landmark coordinates.

### B. Voxelize STL Meshes
```bash
python src/voxelize.py --workers 4
```
Generates 101 aligned binary 3D masks in `masks/`.

### C. Run Classical Baseline
```bash
python src/classical_baseline.py val
```
Evaluates HU windowing [200, 1800] + largest connected component against ground-truth masks.

### D. Run Unit Tests (Geometry & Metrics)
```bash
pytest
```
Validates analytical accuracy of Dice, HD95, ASD, sphere fitting, and 3D angle calculations across test suite in `tests/`.
