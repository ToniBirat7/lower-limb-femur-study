# Data Card: HFValid Femur Collection v3

## 1. Overview
- **Dataset Title:** HFValid collection: Hip-Fracture validation collection (v3)
- **Institution:** IRCCS Istituto Ortopedico Rizzoli (IOR) & Alma Mater Studiorum - University of Bologna, Italy
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Publication Year:** 2025
- **DOI:** [10.6092/unibo/amsacta/8431](https://doi.org/10.6092/unibo/amsacta/8431)
- **Subjects:** 101 whole-femur CT scans with ground-truth surface segmentations and anatomical landmarks.

---

## 2. Patient Cohort & Clinical Eligibility
- **Biological Sex:** Female (100%)
- **Age:** 55 to 88 years ($72.4 \pm 7.1$ years)
- **Inclusion Criteria:**
  - Collected at IOR between 1999 and 2016 for contralateral hip arthroplasty surgical planning.
  - Followed for minimum 5 years to observe fracture occurrence.
- **Exclusion Criteria:**
  - Autoimmune diseases (e.g. rheumatoid arthritis), metastatic pathologies.
  - Pre-existing prosthetic implants or metallic hardware.
  - Severe deformities in the modeled femur.

---

## 3. Imaging Acquisition & Resolution
- **Scanner Models:** GE Medical Systems (BrightSpeed, HiSpeed CT/i, LightSpeed16)
- **Tube Voltage (kVp):** 120 kVp
- **Tube Current:** 140 to 220 mA
- **Slice Thickness:** 2.0 mm to 5.0 mm (mean $2.43$ mm)
- **In-Plane Spacing ($dX, dY$):** $0.586$ mm to $0.977$ mm (mean $0.764$ mm, median $0.742$ mm)
- **Voxel Dimensions:**
  - $X$: 230 to 256 voxels (median 256)
  - $Y$: 236 to 472 voxels (median 324)
  - $Z$: 201 to 494 voxels (median 273)
- **Radiological Calibration:** Asynchronously calibrated via European Spine Phantom (ESP) scans providing density conversion: $\text{Density (mg/cm}^3\text{)} = a + b \times \text{HU}$.

---

## 4. Ground-Truth Reference Annotations
Every subject folder contains three synchronized modalities:
1. `PatXXX.nrrd`: 3D calibrated CT volume.
2. `PatXXX.stl`: Reference 3D surface mesh extracted by orthopaedic engineering specialists.
3. `PatXXX_palp.inp`: Standardized 3D landmark coordinates:
   - `femurHead`: Femoral head center $(X, Y, Z)$
   - `headDir`: Femoral neck direction vector
   - `greatTroch`: Greater trochanter tip
   - `kneeCenter`: Distal intercondylar notch / knee hinge center

---

## 5. Dataset Partitioning (Patient-Level)
To guarantee zero data leakage between training, validation, and evaluation, a deterministic patient-level split (stratified by anatomical laterality) was generated:
- **Training Set:** 71 subjects (70.3%)
- **Validation Set:** 10 subjects (9.9%)
- **Test Set:** 20 subjects (19.8%) - **Held out, evaluated strictly once.**
