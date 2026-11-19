"""
How much does leg rotation change the neck-shaft angle (NSA) seen on a front-view radiograph?

For each femur, the 3D neck and shaft axes from `planning.py` are expressed in a
femoral frame (z: mechanical axis, x: posterior condylar axis, y: anterior).
The femur is rotated about the mechanical axis (positive = internal rotation)
and both axes are projected onto the detector plane of a parallel-beam
front-view radiograph. The projected NSA is compared with the 3D NSA.

Expected from projection geometry: with the knee facing forward, an anteverted
neck is foreshortened and looks more valgus (larger NSA); the error is smallest
when the leg is internally rotated by about the anteversion angle.

    python src/xray_projection.py --split all
    python src/xray_projection.py --figure Pat003
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR, landmarks_path, load_sides, load_splits, stl_path
from src.planning import load_mesh, measure, parse_palp_landmarks, unit

ROTATIONS = np.arange(-30, 31, 5)


def rotate_vector_around_axis(vector, axis, angle_degrees):
    """Rodrigues rotation of `vector` about `axis`."""
    rad = np.radians(angle_degrees)
    k = axis / np.linalg.norm(axis)
    v = np.asarray(vector, dtype=np.float64)
    return v * np.cos(rad) + np.cross(k, v) * np.sin(rad) + k * np.dot(k, v) * (1.0 - np.cos(rad))


def femoral_frame(mech_up, condylar_axis):
    """Rows: lateral (x), anterior (y), up (z), as unit vectors in CT space."""
    z = unit(mech_up)
    x = unit(condylar_axis - (condylar_axis @ z) * z)
    y = np.cross(z, x)
    # choose the sign so that y points anteriorly (LPS: anterior is -y)
    if y @ np.array([0.0, -1.0, 0.0]) < 0:
        y = -y
    return np.vstack([x, y, z])


def projected_nsa(neck_f, shaft_f, rotation_deg):
    """NSA on the detector after rotating the femur internally by `rotation_deg`.

    Vectors are in femoral-frame components. Internal rotation turns the lateral
    axis toward anterior, which reduces the posterior tilt of an anteverted neck.
    """
    t = np.radians(rotation_deg)
    rot = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    n, s = rot @ neck_f, rot @ shaft_f
    n2, s2 = n[[0, 2]], s[[0, 2]]  # detector plane: lateral-up
    cos = n2 @ s2 / (np.linalg.norm(n2) * np.linalg.norm(s2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def run(cases):
    sides = load_sides()
    rows = []
    for cid in cases:
        lm = parse_palp_landmarks(landmarks_path(cid))
        try:
            m = measure(load_mesh(stl_path(cid)), sides[cid], lm["kneeCenter"])
        except Exception as e:
            print(f"  {cid}: failed ({e})")
            continue
        if m["neck_axis"] is None or m["condylar_axis"] is None:
            print(f"  {cid}: no neck or condylar axis")
            continue
        frame = femoral_frame(m["mech_up"], m["condylar_axis"])
        neck_f, shaft_f = frame @ m["neck_axis"], frame @ m["shaft_up"]
        for r in ROTATIONS:
            nsa2d = projected_nsa(neck_f, shaft_f, r)
            rows.append({
                "subject_id": cid,
                "anteversion_deg": round(m["anteversion_deg"], 2),
                "nsa_3d_deg": round(m["nsa_deg"], 2),
                "rotation_deg": int(r),
                "nsa_2d_deg": round(nsa2d, 2),
                "error_deg": round(nsa2d - m["nsa_deg"], 2),
            })
        print(f"  {cid}: anteversion={m['anteversion_deg']:.1f}, NSA 3D={m['nsa_deg']:.1f}")
    return pd.DataFrame(rows)


def summarise(df):
    by_rot = df.groupby("rotation_deg")["error_deg"].agg(["mean", "std", "min", "max"]).round(2)
    per_case = df.loc[df.groupby("subject_id")["error_deg"].apply(lambda e: e.abs().idxmin())]
    r = np.corrcoef(per_case["anteversion_deg"], per_case["rotation_deg"])[0, 1]
    return by_rot, per_case, r


def plot(df, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_rot = df.groupby("rotation_deg")["error_deg"]
    mean, std = by_rot.mean(), by_rot.std()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.fill_between(mean.index, mean - std, mean + std, alpha=0.25, color="C0", label="mean $\\pm$ 1 SD")
    ax.plot(mean.index, mean.values, "o-", color="C0", label="mean error")
    ax.set_xlabel("leg rotation (degrees, + internal / - external)")
    ax.set_ylabel("projected NSA - 3D NSA (degrees)")
    ax.set_title(f"Neck-shaft angle on a simulated front-view radiograph (n = {df.subject_id.nunique()})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def drr_figure(cid, out_png, rotations=(-20, 0, 20)):
    """Parallel-beam projections of the CT through the femur at three rotations about the CT z axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import SimpleITK as sitk
    from scipy.ndimage import rotate

    from src.config import ct_path

    img = sitk.ReadImage(str(ct_path(cid)))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # z, y, x
    sx, sy, sz = img.GetSpacing()
    # attenuation relative to water, with bone weighted up as in a low-kV radiograph
    mu = np.clip(vol + 1000.0, 0, None) / 1000.0 + 3.0 * np.clip(vol - 200.0, 0, None) / 1000.0
    fig, axes = plt.subplots(1, len(rotations), figsize=(3.2 * len(rotations), 6))
    for ax, r in zip(axes, rotations):
        rv = rotate(mu, r, axes=(1, 2), reshape=False, order=1)
        proj = rv.sum(axis=1) * sy  # integrate along anterior-posterior
        lo, hi = np.percentile(proj, [50, 99.8])
        ax.imshow(np.clip((proj - lo) / (hi - lo), 0, 1)[::-1], cmap="gray", aspect=sz / sx)
        ax.set_title(f"{r:+d} deg")
        ax.axis("off")
    fig.suptitle(f"{cid}: simulated front-view radiographs at three rotations")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="2D versus 3D neck-shaft angle under rotation")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--figure", help="subject ID for the projection image figure")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(exist_ok=True)
    if args.figure:
        drr_figure(args.figure, FIGURES_DIR / f"drr_{args.figure}.png")
        return

    splits = load_splits()
    cases = splits["train"] + splits["val"] + splits["test"] if args.split == "all" else splits[args.split]
    df = run(cases)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / f"xray_projection_{args.split}.csv", index=False)
    by_rot, per_case, r = summarise(df)
    by_rot.to_csv(RESULTS_DIR / f"xray_projection_{args.split}_summary.csv")
    plot(df, FIGURES_DIR / "xray_projection_error.png")
    print("\nError (projected - 3D NSA) by rotation:")
    print(by_rot.to_string())
    print(f"\nRotation with the smallest error vs anteversion: r = {r:.2f} (n = {len(per_case)})")


if __name__ == "__main__":
    main()
