# Data card: HFValid v3

Values below are computed from `HFValid_Metadata.csv` and the image headers (`results/data/hfvalid_audit.csv`, `results/data/data_quality.csv`).

## Source

- HFValid collection v3, IRCCS Istituto Ortopedico Rizzoli and University of Bologna.
- DOI: [10.6092/unibo/amsacta/8431](https://doi.org/10.6092/unibo/amsacta/8431). Licence: CC BY 4.0.
- Inclusion and exclusion criteria are given on the dataset page.
- No images, masks or meshes are stored in this repository.

## Cohort

| Item | Value |
|---|---|
| Subjects | 101, all female |
| Age | 68.6 ± 8.8 years (56 to 86) |
| Height, weight | 160 ± 7 cm, 65 ± 12 kg |
| Scanned side | 64 left, 37 right |
| Later hip fracture | 4 of 101 |

## Imaging

| Item | Value |
|---|---|
| Scanners | GE BrightSpeed (38), GE HiSpeed CT/i (34), Philips Brilliance16 (29) |
| Tube voltage | 120 kVp (74), 140 kVp (27) |
| Tube current | 102 to 200 mA |
| In-plane spacing | 0.764 ± 0.061 mm (0.586 to 0.977) |
| Slice thickness | 2.5 mm (38), 3.0 mm (62), 5.0 mm (1) |
| Spacing between slices | 1.0 to 2.0 mm (mean 1.83) |
| Volume size | 230 to 256 x 236 to 472 x 201 to 494 voxels |

Every scan is an overlapping reconstruction: slices are thicker than the step between them.
This smooths the image along z, so through-plane detail is coarser than the voxel spacing suggests.

Background values differ by scan: 68 scans use -3024 outside the reconstruction circle, 33 use -1024.
Each scan has its own phantom calibration (`coeff a`, `coeff b`) for converting HU to bone density; this study works in HU only.

## Per-subject files

- `PatXXX.nrrd`: CT volume (LPS).
- `PatXXX.stl`: reference femur surface, in the CT frame.
- `PatXXX_palp.inp`: four landmarks that define the dataset's femur frame:
  - `femurHead`: centre of the femoral head (frame origin).
  - `headDir`: a point on the line from the head centre towards the knee (frame X axis). It is not a neck direction.
  - `greatTroch`: greater trochanter (sets the Y axis direction).
  - `kneeCenter`: midpoint between the femoral epicondyles.

Reference masks were made by voxelising each STL on its CT grid (`src/voxelize.py`).

## Quality checks (`src/data_quality.py`)

- No metal inside any mask (no voxel above 3000 HU).
- Head landmark depth below the reference surface: 14.0 to 21.9 mm, consistent with a head centre about one radius inside the head.
- Share of mask voxels above 150 HU: 0.42 to 0.82 (mean 0.62); masks follow the bone, and the lower values come from low-density trabecular bone in older subjects.
- No duplicate images.
- One volume outlier: Pat068 (641 ml against a median of 429 ml), kept in the data.

## Split

Patient-level split in `splits.json`: 71 train, 10 validation, 20 test.
Test cases were scored once, after all methods were fixed.
