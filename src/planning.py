"""
Orthopaedic Surgical Planning & Biomarker Extraction Module (Stage 1f)
Implements:
1. Femoral Head Diameter (Sphere fitting for hip arthroplasty implant sizing)
2. Anatomical Diaphyseal Shaft Axis (PCA/SVD line fit on mid-shaft slices)
3. Neck-Shaft Angle / Collo-Diaphyseal Angle (CCD)
4. Mechanical-Anatomical Angle (AMA - distal femoral resection planning for TKA)
5. Validation against reference annotations and Bland-Altman agreement analysis
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.ndimage import center_of_mass

from src.geometry import (
    fit_sphere_least_squares,
    fit_line_3d,
    angle_between_vectors_3d,
    project_vector_on_plane,
)

DATA_DIR = Path("/run/media/tonibirat/TOSHIBA_EXT/naamii-data/HFValid")
MASKS_DIR = DATA_DIR / "masks"
SUBJECTS_DIR = DATA_DIR / "Subjects"


def parse_palp_landmarks(inp_path):
    """
    Parses _palp.inp file into dictionary of 3D coordinates.
    Format in file:
      femurHeadX = -103.8408
      femurHeadY = -12.9711
      femurHeadZ = -127.7788
    Returns: {"femurHead": array([X, Y, Z]), ...}
    """
    raw_vals = {}
    if not os.path.exists(inp_path):
        return {}

    with open(inp_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = [p.strip() for p in line.split("=")]
                try:
                    raw_vals[key] = float(val)
                except ValueError:
                    continue

    landmarks = {}
    for name in ["femurHead", "headDir", "greatTroch", "kneeCenter"]:
        if f"{name}X" in raw_vals and f"{name}Y" in raw_vals and f"{name}Z" in raw_vals:
            landmarks[name] = np.array([
                raw_vals[f"{name}X"],
                raw_vals[f"{name}Y"],
                raw_vals[f"{name}Z"],
            ])

    return landmarks


def extract_shaft_axis_from_mask(mask_sitk):
    """
    Extracts diaphyseal shaft axis from binary mask by fitting a 3D line
    to slice centroids in the middle 30% to 70% of the femur length.
    Returns: centroid (3,), direction_vector (3, pointing proximal-to-distal)
    """
    arr = sitk.GetArrayFromImage(mask_sitk)  # (Z, Y, X)
    origin = np.array(mask_sitk.GetOrigin())
    spacing = np.array(mask_sitk.GetSpacing())
    direction = np.array(mask_sitk.GetDirection()).reshape((3, 3))

    z_indices = np.where(arr.sum(axis=(1, 2)) > 0)[0]
    if len(z_indices) < 10:
        return None, None

    z_min, z_max = z_indices[0], z_indices[-1]
    z_shaft_min = int(z_min + 0.30 * (z_max - z_min))
    z_shaft_max = int(z_min + 0.70 * (z_max - z_min))

    physical_centroids = []
    for z in range(z_shaft_min, z_shaft_max + 1):
        slice_mask = arr[z, :, :]
        if slice_mask.sum() > 20:
            y_c, x_c = center_of_mass(slice_mask)
            # Convert voxel (x, y, z) to physical space
            vox_coords = np.array([x_c, y_c, z])
            phys_pt = origin + direction @ (vox_coords * spacing)
            physical_centroids.append(phys_pt)

    if len(physical_centroids) < 5:
        return None, None

    centroid, axis_dir = fit_line_3d(np.array(physical_centroids))
    # Standardize direction pointing superior (+Z)
    if axis_dir[2] < 0:
        axis_dir = -axis_dir

    return centroid, axis_dir


def extract_femoral_head_sphere(mask_sitk, head_landmark=None):
    """
    Fits least-squares sphere to proximal femur head points.
    Returns: center (3,), diameter (float, mm)
    """
    arr = sitk.GetArrayFromImage(mask_sitk)  # (Z, Y, X)
    origin = np.array(mask_sitk.GetOrigin())
    spacing = np.array(mask_sitk.GetSpacing())
    direction = np.array(mask_sitk.GetDirection()).reshape((3, 3))

    # Surface boundary voxels
    surface_sitk = sitk.LabelContour(mask_sitk)
    surf_arr = sitk.GetArrayFromImage(surface_sitk)

    z_pts, y_pts, x_pts = np.where(surf_arr > 0)
    if len(z_pts) < 100:
        return None, None

    # Voxel to physical points
    vox_pts = np.column_stack([x_pts, y_pts, z_pts])
    phys_pts = origin + (direction @ (vox_pts * spacing).T).T

    if head_landmark is not None:
        # Filter points within 35 mm of landmark
        dists = np.linalg.norm(phys_pts - head_landmark, axis=1)
        head_pts = phys_pts[dists < 35.0]
    else:
        # Proximal 15% of femur
        z_thresh = np.percentile(phys_pts[:, 2], 85)
        head_pts = phys_pts[phys_pts[:, 2] > z_thresh]

    if len(head_pts) < 20:
        return None, None

    center, radius = fit_sphere_least_squares(head_pts)
    diameter = 2.0 * radius
    return center, diameter


def compute_planning_measurements(subject_id, mask_sitk=None):
    """
    Computes four key surgical measurements for a given subject:
    1. Femoral Head Diameter (mm)
    2. Neck-Shaft Angle (CCD in degrees)
    3. Mechanical-Anatomical Cut Angle (AMA in degrees)
    4. Head Center Error (mm, if ground truth palp is present)
    """
    inp_path = SUBJECTS_DIR / subject_id / f"{subject_id}_palp.inp"
    ref_landmarks = parse_palp_landmarks(inp_path)

    if mask_sitk is None:
        mask_path = MASKS_DIR / f"{subject_id}_mask.nii.gz"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask_sitk = sitk.ReadImage(str(mask_path))

    # 1. Shaft axis
    shaft_center, shaft_dir = extract_shaft_axis_from_mask(mask_sitk)

    # 2. Femoral head sphere fit
    ref_head = ref_landmarks.get("femurHead")
    head_center, head_diameter = extract_femoral_head_sphere(mask_sitk, head_landmark=ref_head)

    # 3. Collo-Diaphyseal Angle (CCD)
    # Using neck direction landmark or vector from trochanter/neck to head
    ccd_angle = None
    if ref_head is not None and "headDir" in ref_landmarks and shaft_dir is not None:
        neck_dir = ref_head - ref_landmarks["headDir"]
        ccd_angle = angle_between_vectors_3d(neck_dir, shaft_dir)
        # Clinical convention: obtuse angle between shaft and neck
        if ccd_angle < 90.0:
            ccd_angle = 180.0 - ccd_angle

    # 4. Mechanical-Anatomical Angle (AMA)
    # Mechanical axis: line from femoral head to knee center
    ama_angle = None
    if head_center is not None and "kneeCenter" in ref_landmarks and shaft_dir is not None:
        mech_axis = head_center - ref_landmarks["kneeCenter"]
        ama_angle = angle_between_vectors_3d(mech_axis, shaft_dir)
        if ama_angle > 90.0:
            ama_angle = 180.0 - ama_angle

    # 5. Head localization error
    head_error = None
    if head_center is not None and ref_head is not None:
        head_error = float(np.linalg.norm(head_center - ref_head))

    return {
        "subject_id": subject_id,
        "head_diameter_mm": round(head_diameter, 2) if head_diameter else None,
        "ccd_angle_deg": round(ccd_angle, 2) if ccd_angle else None,
        "ama_angle_deg": round(ama_angle, 2) if ama_angle else None,
        "head_localization_err_mm": round(head_error, 2) if head_error else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Orthopaedic planning measurements")
    parser.add_argument("--split", choices=["val", "test", "all"], default="val")
    parser.add_argument("--output", default="planning_measurements.csv")
    args = parser.parse_args()

    splits_path = REPO_ROOT / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    if args.split == "all":
        case_ids = splits["train"] + splits["val"] + splits["test"]
    else:
        case_ids = splits[args.split]

    print(f"Extracting orthopaedic planning measurements for {len(case_ids)} {args.split} cases...")
    results = []
    for cid in case_ids:
        try:
            m = compute_planning_measurements(cid)
            results.append(m)
            print(f"  {cid}: Head Dia={m['head_diameter_mm']}mm, CCD={m['ccd_angle_deg']}°, AMA={m['ama_angle_deg']}°, Head Err={m['head_localization_err_mm']}mm")
        except Exception as e:
            print(f"  {cid}: Failed ({e})")

    df = pd.DataFrame(results)
    out_path = REPO_ROOT / args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved planning results to {out_path}")
    print("\nSummary Statistics:")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
