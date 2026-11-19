"""
Inter-subject femur registration, scored by target registration error (TRE)
at the dataset landmarks (head centre, greater trochanter, knee centre).

Methods, all on the same 10 same-side pairs:
- initial: centre-of-mass alignment of the bone masks
- rigid image registration on CT with Mattes mutual information (whole image;
  restricting the metric to a femur mask broke convergence in testing)
- rigid and affine registration of smoothed bone masks (mean squares)
- rigid ICP between the surface meshes (VTK)

Between different people part of the remaining error is real anatomical
difference, so the numbers compare methods rather than measure an accuracy limit.
A second test resamples one CT with a known rigid transform and registers it
back, where the remaining error is registration error alone.

    python src/registration.py --pairs 10
    python src/registration.py --synthetic 5
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import vtk

from src.config import RESULTS_DIR, ct_path, landmarks_path, load_sides, load_splits, mask_path, stl_path
from src.planning import parse_palp_landmarks

LANDMARKS = ["femurHead", "greatTroch", "kneeCenter"]


def tre(transform_point, fixed_lm, moving_lm):
    """Mean distance (mm) between transformed fixed landmarks and moving landmarks."""
    return float(np.mean([np.linalg.norm(np.array(transform_point(fixed_lm[k])) - moving_lm[k]) for k in LANDMARKS]))


def image_registration(fixed, moving, initial, metric="mask", fixed_mask=None):
    """Optimise `initial` (Euler3D or Affine). metric: "mi" on CT, or "mask" (mean squares on smoothed bone masks)."""
    reg = sitk.ImageRegistrationMethod()
    if metric == "mi":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1, seed=0)
    if fixed_mask is not None:
        reg.SetMetricFixedMask(fixed_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
                                                 numberOfIterations=300, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    tx = reg.Execute(fixed, moving)
    return tx.GetNthTransform(0) if isinstance(tx, sitk.CompositeTransform) else tx


def smooth_mask(cid):
    return sitk.SmoothingRecursiveGaussian(sitk.Cast(sitk.ReadImage(str(mask_path(cid))) > 0, sitk.sitkFloat32), 2.0)


def to_affine(rigid):
    rigid = sitk.Euler3DTransform(rigid)
    a = sitk.AffineTransform(3)
    a.SetCenter(rigid.GetCenter())
    a.SetMatrix(rigid.GetMatrix())
    a.SetTranslation(rigid.GetTranslation())
    return a


def read_ct(cid):
    img = sitk.ReadImage(str(ct_path(cid)), sitk.sitkFloat32)
    return sitk.Clamp(img, sitk.sitkFloat32, -200.0, 1500.0)


def icp(fixed_cid, moving_cid):
    """Rigid ICP mapping the fixed surface onto the moving surface (same direction as SimpleITK)."""
    def read(cid):
        r = vtk.vtkSTLReader()
        r.SetFileName(str(stl_path(cid)))
        r.Update()
        dec = vtk.vtkQuadricDecimation()
        dec.SetInputData(r.GetOutput())
        dec.SetTargetReduction(0.9)
        dec.Update()
        return dec.GetOutput()

    src, tgt = read(fixed_cid), read(moving_cid)
    # start from centroid alignment
    c_src = np.mean(np.array([src.GetPoint(i) for i in range(src.GetNumberOfPoints())]), axis=0)
    c_tgt = np.mean(np.array([tgt.GetPoint(i) for i in range(tgt.GetNumberOfPoints())]), axis=0)
    shift = vtk.vtkTransform()
    shift.Translate(*(c_tgt - c_src))
    tf = vtk.vtkTransformFilter()
    tf.SetInputData(src)
    tf.SetTransform(shift)
    tf.Update()

    icp_tx = vtk.vtkIterativeClosestPointTransform()
    icp_tx.SetSource(tf.GetPolyDataOutput())
    icp_tx.SetTarget(tgt)
    icp_tx.GetLandmarkTransform().SetModeToRigidBody()
    icp_tx.SetMaximumNumberOfIterations(200)
    icp_tx.SetMaximumNumberOfLandmarks(2000)
    icp_tx.StartByMatchingCentroidsOff()
    icp_tx.Modified()
    icp_tx.Update()

    def apply(p):
        return icp_tx.TransformPoint(shift.TransformPoint(tuple(map(float, p))))
    return apply


def make_pairs(n):
    """Fixed from validation, moving from test, same side, in a fixed order."""
    splits, sides = load_splits(), load_sides()
    pairs = []
    for f in splits["val"]:
        for m in splits["test"]:
            if sides[f] == sides[m] and m not in [p[1] for p in pairs]:
                pairs.append((f, m))
                break
    for f in splits["val"]:  # second pass if fewer than n pairs
        for m in splits["test"]:
            if len(pairs) >= n:
                break
            if sides[f] == sides[m] and (f, m) not in pairs and m not in [p[1] for p in pairs]:
                pairs.append((f, m))
    return pairs[:n]


def run_pair(f, m):
    lf, lm = parse_palp_landmarks(landmarks_path(f)), parse_palp_landmarks(landmarks_path(m))
    fixed_mask = sitk.ReadImage(str(mask_path(f))) > 0
    moving_mask = sitk.ReadImage(str(mask_path(m))) > 0
    init = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_mask, sitk.sitkFloat32), sitk.Cast(moving_mask, sitk.sitkFloat32),
        sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    row = {"fixed": f, "moving": m,
           "identity_mm": tre(lambda p: p, lf, lm),
           "initial_mm": tre(init.TransformPoint, lf, lm)}

    rigid_mi = image_registration(read_ct(f), read_ct(m), sitk.Euler3DTransform(init), "mi")
    row["rigid_mi_mm"] = tre(rigid_mi.TransformPoint, lf, lm)

    sf, sm = smooth_mask(f), smooth_mask(m)
    rigid = image_registration(sf, sm, sitk.Euler3DTransform(init), "mask")
    row["rigid_mask_mm"] = tre(rigid.TransformPoint, lf, lm)
    affine = image_registration(sf, sm, to_affine(rigid), "mask")
    row["affine_mask_mm"] = tre(affine.TransformPoint, lf, lm)

    row["icp_mm"] = tre(icp(f, m), lf, lm)
    return row


def synthetic_recovery(cid, rotation_deg=8.0, shift_mm=(12.0, -8.0, 15.0)):
    """Resample a CT with a known rigid transform and register it back (same subject).

    Here the true answer is known, so the remaining TRE is registration error alone.
    """
    fixed = read_ct(cid)
    truth = sitk.Euler3DTransform()
    truth.SetCenter(fixed.TransformContinuousIndexToPhysicalPoint([s / 2.0 for s in fixed.GetSize()]))
    truth.SetRotation(np.radians(rotation_deg), np.radians(-rotation_deg / 2), np.radians(rotation_deg / 2))
    truth.SetTranslation(shift_mm)
    moving = sitk.Resample(fixed, fixed, truth.GetInverse(), sitk.sitkLinear, -200.0)
    lm = parse_palp_landmarks(landmarks_path(cid))
    # moving(x) = fixed(truth^-1(x)), so a fixed point p appears at truth(p) in the moving image
    moved_lm = {k: np.array(truth.TransformPoint(tuple(lm[k]))) for k in LANDMARKS}

    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    est = image_registration(fixed, moving, sitk.Euler3DTransform(init), "mi")
    return {"subject": cid, "before_mm": tre(lambda p: p, lm, moved_lm),
            "after_mm": tre(est.TransformPoint, lm, moved_lm)}


def main():
    parser = argparse.ArgumentParser(description="Femur registration with landmark TRE")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--synthetic", type=int, default=0, help="run the known-transform test on N subjects")
    args = parser.parse_args()

    if args.synthetic:
        rows = [synthetic_recovery(c) for c in load_splits()["val"][: args.synthetic]]
        df = pd.DataFrame(rows)
        RESULTS_DIR.mkdir(exist_ok=True)
        df.to_csv(RESULTS_DIR / "registration_synthetic.csv", index=False)
        print(df.round(2).to_string(index=False))
        print(df[["before_mm", "after_mm"]].agg(["mean", "max"]).round(2).to_string())
        return

    rows = []
    for f, m in make_pairs(args.pairs):
        row = run_pair(f, m)
        rows.append(row)
        print(", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "registration_tre.csv", index=False)
    cols = [c for c in df.columns if c.endswith("_mm")]
    print("\nTRE (mm), mean over landmarks then over pairs:")
    print(df[cols].agg(["mean", "std", "median"]).round(2).to_string())


if __name__ == "__main__":
    main()
