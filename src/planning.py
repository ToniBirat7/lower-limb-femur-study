"""
Femoral planning measurements from a surface mesh.

All axes are found from the bone surface; only the knee centre comes from the
dataset landmarks (it is defined from the epicondyles, which a single surface
does not identify reliably). The same code runs on reference meshes and on
meshes extracted from predicted masks, so the two can be compared.

Measurements
- head centre and diameter: trimmed least-squares sphere fit
- neck axis: line through the centroids of cross-sections of the neck
- shaft axis: line through the centroids of cross-sections at 30 to 70 percent of femur length
- neck-shaft angle (NSA): angle between neck axis and shaft axis
- anteversion: angle between neck axis and posterior condylar axis, viewed along the mechanical axis
- mechanical-anatomical angle (AMA): angle between mechanical axis and shaft axis

Coordinates are in the CT frame, which is LPS (x to patient left, y posterior, z superior).

    python src/planning.py --split val            # reference meshes
    python src/planning.py --split test --method nnunet
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pyvista as pv

from src.config import RESULTS_DIR, landmarks_path, load_sides, load_splits, mask_path, prediction_path, stl_path
from src.geometry import fit_line_3d, fit_sphere_least_squares

UP = np.array([0.0, 0.0, 1.0])
POSTERIOR = np.array([0.0, 1.0, 0.0])


def parse_palp_landmarks(inp_path):
    """Read a HFValid `_palp.inp` file into {name: xyz}."""
    raw = {}
    if not os.path.exists(inp_path):
        return {}
    with open(inp_path) as f:
        for line in f:
            if "=" in line:
                key, val = (p.strip() for p in line.split("="))
                try:
                    raw[key] = float(val)
                except ValueError:
                    pass
    out = {}
    for name in ["femurHead", "headDir", "greatTroch", "kneeCenter"]:
        if all(f"{name}{a}" in raw for a in "XYZ"):
            out[name] = np.array([raw[f"{name}{a}"] for a in "XYZ"])
    return out


def unit(v):
    return v / np.linalg.norm(v)


def lateral_direction(side):
    """Lateral unit vector in LPS: a right femur's lateral side is patient right (-x)."""
    return np.array([-1.0, 0.0, 0.0]) if side == "R" else np.array([1.0, 0.0, 0.0])


def fit_femoral_head(points, side):
    """Trimmed sphere fit seeded at the highest point on the medial side of the proximal femur."""
    z = points[:, 2]
    proximal = points[z > z.max() - 70.0]
    medial = -lateral_direction(side)
    med_score = proximal @ medial
    candidates = proximal[med_score > np.median(med_score)]
    seed = candidates[np.argmax(candidates[:, 2])]

    center, radius = seed - np.array([0.0, 0.0, 22.0]), 22.0
    near = points[np.linalg.norm(points - seed, axis=1) < 40.0]
    for _ in range(10):
        d = np.linalg.norm(near - center, axis=1)
        keep = near[np.abs(d - radius) < max(2.0, 0.15 * radius)]
        if len(keep) < 50:
            break
        new_center, new_radius = fit_sphere_least_squares(keep)
        if np.linalg.norm(new_center - center) < 1e-3:
            center, radius = new_center, new_radius
            break
        center, radius = new_center, new_radius
    return center, radius


def section_centroid(mesh, origin, normal, max_dist=35.0):
    """Centroid of the cross-section loop closest to `origin` on the plane (origin, normal)."""
    section = mesh.slice(normal=normal, origin=origin)
    if section.n_points == 0:
        return None
    loops = section.connectivity()
    best, best_d = None, np.inf
    for rid in np.unique(loops.point_data["RegionId"]):
        pts = loops.points[loops.point_data["RegionId"] == rid]
        c = pts.mean(axis=0)
        d = np.linalg.norm(c - origin)
        if d < best_d:
            best, best_d = c, d
    return best if best_d < max_dist else None


def narrowest_section(mesh, head_center, axis, head_radius):
    """Centroid of the smallest neck cross-section perpendicular to `axis`."""
    best, best_r = None, np.inf
    for s in np.linspace(0.7 * head_radius, 1.7 * head_radius, 12):
        origin = head_center + s * axis
        section = mesh.slice(normal=axis, origin=origin)
        if section.n_points == 0:
            continue
        loops = section.connectivity()
        for rid in np.unique(loops.point_data["RegionId"]):
            pts = loops.points[loops.point_data["RegionId"] == rid]
            c = pts.mean(axis=0)
            if np.linalg.norm(c - origin) > 15.0:
                continue
            r = np.linalg.norm(pts - c, axis=1).mean()
            if r < best_r:
                best, best_r = c, r
    return best


def fit_neck_axis(mesh, head_center, head_radius, side):
    """Neck axis: head centre to the centroid of the narrowest neck section, iterated to a fixed point."""
    axis = unit(lateral_direction(side) - 0.8 * UP)
    for _ in range(30):
        c = narrowest_section(mesh, head_center, axis, head_radius)
        if c is None:
            return None
        new = unit(c - head_center)
        step = np.degrees(np.arccos(np.clip(new @ axis, -1.0, 1.0)))
        axis = unit(axis + new)  # damped update
        if step < 0.1:
            return new
    return axis


def fit_shaft_axis(mesh, head_center, knee_center):
    """Line through section centroids between 30 and 70 percent of the head-to-knee distance."""
    mech = knee_center - head_center
    length = np.linalg.norm(mech)
    down = mech / length
    cents = []
    for frac in np.linspace(0.30, 0.70, 15):
        origin = head_center + frac * mech
        # slice perpendicular to the CT z axis keeps sections closed on the shaft
        # the shaft can sit 60 mm from the head-knee line in femurs with a large offset
        c = section_centroid(mesh, origin, UP, max_dist=80.0)
        if c is not None:
            cents.append(c)
    _, d = fit_line_3d(np.array(cents))
    return d if d @ down < 0 else -d  # points proximally


def posterior_condylar_axis(points, knee_center, mech_up, side):
    """Medial-to-lateral line through the most posterior point of each condyle."""
    h = (points - knee_center) @ mech_up
    distal = points[(h > -40.0) & (h < 15.0)]
    lat = lateral_direction(side)
    lat = unit(lat - (lat @ mech_up) * mech_up)
    post = unit(POSTERIOR - (POSTERIOR @ mech_up) * mech_up)
    m = (distal - knee_center) @ lat
    medial_pts, lateral_pts = distal[m < -5.0], distal[m > 5.0]
    if len(medial_pts) == 0 or len(lateral_pts) == 0:
        return None
    p_med = medial_pts[np.argmax(medial_pts @ post)]
    p_lat = lateral_pts[np.argmax(lateral_pts @ post)]
    return unit(p_lat - p_med)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1.0, 1.0))))


def in_plane(v, normal):
    n = unit(normal)
    return v - (v @ n) * n


def measure(mesh, side, knee_center):
    points = np.asarray(mesh.points)
    head_c, head_r = fit_femoral_head(points, side)
    neck = fit_neck_axis(mesh, head_c, head_r, side)
    shaft_up = fit_shaft_axis(mesh, head_c, knee_center)
    mech_up = unit(head_c - knee_center)
    condyl = posterior_condylar_axis(points, knee_center, mech_up, side)

    out = {
        "head_center": head_c,
        "head_diameter_mm": 2.0 * head_r,
        "ama_deg": angle_deg(mech_up, shaft_up),
        "neck_axis": neck,
        "shaft_up": shaft_up,
        "mech_up": mech_up,
        "condylar_axis": condyl,
        "nsa_deg": np.nan,
        "anteversion_deg": np.nan,
    }
    if neck is not None:
        # neck axis points from the head toward the lateral side; the NSA is the
        # angle between it and the shaft pointing proximally (about 125-135 deg)
        out["nsa_deg"] = angle_deg(neck, shaft_up)
        if condyl is not None:
            n2 = unit(in_plane(neck, mech_up))
            c2 = unit(in_plane(condyl, mech_up))
            ang = angle_deg(n2, c2)
            # positive when the head points anteriorly (head side of the neck is anterior)
            anterior = -unit(in_plane(POSTERIOR, mech_up))
            sign = 1.0 if (-n2) @ anterior > 0 else -1.0
            out["anteversion_deg"] = sign * ang
    return out


def load_mesh(source):
    """`source` is an STL path or a binary mask image (marching cubes on the mask)."""
    source = Path(source)
    if source.suffix == ".stl":
        return pv.read(str(source))
    import SimpleITK as sitk

    img = sitk.ReadImage(str(source))
    if not np.allclose(np.array(img.GetDirection()), np.eye(3).ravel()):
        raise ValueError(f"non-identity direction in {source}")
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0).astype(np.uint8)
    grid = pv.ImageData(dimensions=arr.shape, spacing=img.GetSpacing(), origin=img.GetOrigin())
    grid.point_data["m"] = arr.ravel(order="F")
    surf = grid.contour([0.5], scalars="m").extract_largest().clean()
    # voxel steps make neck sections ragged and the neck fit unstable; Taubin smoothing
    # removes them without shrinking the surface
    return surf.smooth_taubin(n_iter=50, pass_band=0.05)


def run(cases, method):
    sides = load_sides()
    rows = []
    for cid in cases:
        lm = parse_palp_landmarks(landmarks_path(cid))
        if method == "reference":
            src = stl_path(cid)
        elif method == "reference_mask":  # control: reference mask through the same mask-to-mesh path
            src = mask_path(cid)
        else:
            src = prediction_path(method, cid)
        if not Path(src).exists():
            print(f"  {cid}: missing {src}")
            continue
        try:
            m = measure(load_mesh(src), sides[cid], lm["kneeCenter"])
        except Exception as e:  # report and continue with the other cases
            print(f"  {cid}: failed ({e})")
            continue
        rows.append({
            "subject_id": cid,
            "side": sides[cid],
            "head_diameter_mm": round(m["head_diameter_mm"], 2),
            "head_center_err_mm": round(float(np.linalg.norm(m["head_center"] - lm["femurHead"])), 2),
            "nsa_deg": round(m["nsa_deg"], 2),
            "anteversion_deg": round(m["anteversion_deg"], 2),
            "ama_deg": round(m["ama_deg"], 2),
        })
        print(f"  {cid}: " + ", ".join(f"{k}={v}" for k, v in rows[-1].items() if k != "subject_id"))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Femoral planning measurements")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--method", default="reference", help="reference (STL), reference_mask, or a prediction folder name")
    args = parser.parse_args()

    splits = load_splits()
    cases = splits["train"] + splits["val"] + splits["test"] if args.split == "all" else splits[args.split]
    df = run(cases, args.method)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"planning_{args.method}_{args.split}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(df.drop(columns=["subject_id", "side"]).describe().round(2).to_string())


if __name__ == "__main__":
    main()
