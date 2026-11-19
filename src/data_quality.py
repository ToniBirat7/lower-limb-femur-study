"""
Data quality checks on all 101 subjects, run before any model is trained.

Per subject:
- intensity range consistent with calibrated HU
- reference mask aligned with bone: share of mask voxels above 150 HU
  (flagged below 0.3; low-density trabecular bone makes values near 0.6 normal)
- metal: voxels above 3000 HU inside the mask
- head landmark plausibility: distance from the femurHead landmark to the reference surface
  (a head centre should sit about one head radius, ~20 mm, inside the surface)
- duplicate scans: identical image hashes
- mask volume outliers (robust z-score)

    python src/data_quality.py
"""

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pyvista as pv
import SimpleITK as sitk

from src.config import RESULTS_DIR, ct_path, landmarks_path, load_splits, mask_path, stl_path
from src.planning import parse_palp_landmarks


def check_subject(cid):
    ct = sitk.ReadImage(str(ct_path(cid)))
    arr = sitk.GetArrayFromImage(ct)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path(cid)))) > 0
    voxel_ml = np.prod(ct.GetSpacing()) / 1000.0
    lm = parse_palp_landmarks(landmarks_path(cid))
    surface = pv.read(str(stl_path(cid)))
    head_depth = surface.find_closest_point(lm["femurHead"])
    head_depth = float(np.linalg.norm(surface.points[head_depth] - lm["femurHead"]))

    inside = arr[mask]
    return {
        "subject_id": cid,
        "size": "x".join(map(str, ct.GetSize())),
        "spacing_mm": "x".join(f"{s:.3f}" for s in ct.GetSpacing()),
        "hu_min": int(arr.min()),
        "hu_max": int(arr.max()),
        "mask_ml": round(mask.sum() * voxel_ml, 1),
        "mask_above_150hu": round(float((inside > 150).mean()), 3),
        "metal_voxels": int((inside > 3000).sum()),
        "head_landmark_depth_mm": round(head_depth, 1),
        "hash": hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12],
    }


def main():
    splits = load_splits()
    cases = sorted(splits["train"] + splits["val"] + splits["test"])
    df = pd.DataFrame([check_subject(c) for c in cases])

    vol = df["mask_ml"]
    robust_z = 0.6745 * (vol - vol.median()) / (vol - vol.median()).abs().median()
    flags = []
    for i, r in df.iterrows():
        f = []
        if r.hu_min > -900 or r.hu_max < 1000:
            f.append("unusual HU range")
        if r.mask_above_150hu < 0.3:  # elderly femurs: about 40% of the bone is below 150 HU
            f.append("mask poorly aligned with bone")
        if r.metal_voxels > 50:
            f.append("possible metal")
        if not 10 <= r.head_landmark_depth_mm <= 30:
            f.append("head landmark not inside head")
        if abs(robust_z[i]) > 3.5:
            f.append("mask volume outlier")
        if (df["hash"] == r["hash"]).sum() > 1:
            f.append("duplicate image")
        flags.append("; ".join(f) if f else "ok")
    df["flags"] = flags

    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "data_quality.csv", index=False)
    print(df.drop(columns=["hash"]).describe().round(2).to_string())
    print("\nFlagged subjects:")
    flagged = df[df["flags"] != "ok"][["subject_id", "flags"]]
    print(flagged.to_string(index=False) if len(flagged) else "none")


if __name__ == "__main__":
    main()
