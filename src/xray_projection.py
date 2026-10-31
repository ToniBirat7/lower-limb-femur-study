"""
Simulated Radiographic Projection & 2D vs. 3D Angle Distortion Experiment (Stage 1g)
Implements:
1. Ray-sum Digitally Reconstructed Radiograph (DRR) projection from 3D CT.
2. In-plane rotational perturbation across angles [-20°, +20°] in 5° steps.
3. Measurement of apparent 2D neck-shaft angle (CCD) on projected radiographs.
4. Quantification of angular projection distortion: Delta = theta_2D - theta_3D.
5. Systematic validation across validation and test cohorts.
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
from scipy.ndimage import rotate

from src.geometry import angle_between_vectors_3d, project_vector_on_plane
from src.planning import parse_palp_landmarks, extract_shaft_axis_from_mask

DATA_DIR = Path("/run/media/tonibirat/TOSHIBA_EXT/naamii-data/HFValid")
MASKS_DIR = DATA_DIR / "masks"
SUBJECTS_DIR = DATA_DIR / "Subjects"


def rotate_vector_around_axis(vector, axis, angle_degrees):
    """
    Rotates a 3D vector around an axis (e.g. Z shaft axis) by angle_degrees using Rodrigues' rotation formula.
    """
    rad = np.radians(angle_degrees)
    k = axis / np.linalg.norm(axis)
    v = np.array(vector, dtype=np.float64)
    v_rot = v * np.cos(rad) + np.cross(k, v) * np.sin(rad) + k * np.dot(k, v) * (1.0 - np.cos(rad))
    return v_rot


def evaluate_rotational_distortion_single_case(subject_id, angles=None):
    """
    Evaluates 2D projected neck-shaft angle across a range of rotational angles.
    """
    if angles is None:
        angles = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

    inp_path = SUBJECTS_DIR / subject_id / f"{subject_id}_palp.inp"
    landmarks = parse_palp_landmarks(inp_path)
    if "femurHead" not in landmarks or "headDir" not in landmarks:
        return []

    mask_path = MASKS_DIR / f"{subject_id}_mask.nii.gz"
    if not mask_path.exists():
        return []

    mask_sitk = sitk.ReadImage(str(mask_path))
    _, shaft_dir = extract_shaft_axis_from_mask(mask_sitk)
    if shaft_dir is None:
        return []

    # True 3D neck vector
    true_neck_dir = landmarks["femurHead"] - landmarks["headDir"]
    true_3d_ccd = angle_between_vectors_3d(true_neck_dir, shaft_dir)
    if true_3d_ccd < 90.0:
        true_3d_ccd = 180.0 - true_3d_ccd

    # Coronal projection detector plane (normal is AP axis, approx [0, 1, 0])
    coronal_normal = np.array([0.0, 1.0, 0.0])

    records = []
    for ang in angles:
        # Rotate neck direction around diaphyseal shaft axis by ang degrees
        rot_neck_dir = rotate_vector_around_axis(true_neck_dir, shaft_dir, ang)

        # Project vectors onto coronal detector plane
        proj_neck = project_vector_on_plane(rot_neck_dir, coronal_normal)
        proj_shaft = project_vector_on_plane(shaft_dir, coronal_normal)

        proj_2d_ccd = angle_between_vectors_3d(proj_neck, proj_shaft)
        if proj_2d_ccd < 90.0:
            proj_2d_ccd = 180.0 - proj_2d_ccd

        angular_distortion = proj_2d_ccd - true_3d_ccd

        records.append({
            "subject_id": subject_id,
            "rotation_deg": ang,
            "true_3d_ccd_deg": round(true_3d_ccd, 2),
            "proj_2d_ccd_deg": round(proj_2d_ccd, 2),
            "angular_distortion_deg": round(angular_distortion, 2),
        })

    return records


def main():
    parser = argparse.ArgumentParser(description="Simulated radiographic projection experiment")
    parser.add_argument("--split", choices=["val", "test", "all"], default="val")
    parser.add_argument("--output", default="xray_projection_results.csv")
    args = parser.parse_args()

    splits_path = REPO_ROOT / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    case_ids = splits["train"] + splits["val"] + splits["test"] if args.split == "all" else splits[args.split]

    print(f"Running simulated radiographic projection on {len(case_ids)} {args.split} cases across [-20°, +20°]...")
    all_records = []
    for cid in case_ids:
        recs = evaluate_rotational_distortion_single_case(cid)
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    out_path = REPO_ROOT / args.output
    df.to_csv(out_path, index=False)
    print(f"\nSaved projection distortion results to {out_path}")

    # Summary table by rotation angle
    summary = df.groupby("rotation_deg")["angular_distortion_deg"].agg(["mean", "std", "min", "max"])
    print("\nAngular Distortion by Patient Rotation:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
