"""
TotalSegmentator femur segmentation (no training).

Runs the standard 1.5 mm `total` model with a femur ROI subset and keeps the
femur of the side recorded in the dataset metadata. The ground truth is never
used to choose the side.

    python src/totalseg_pipeline.py --split test
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SimpleITK as sitk

from src.config import NNUNET_RAW, ct_path, load_sides, load_splits, prediction_path

METHOD = "totalseg"


def nifti_input(subject_id):
    """TotalSegmentator reads NIfTI, not NRRD; use the converted copies."""
    for folder in ("imagesTr", "imagesTs"):
        p = NNUNET_RAW / folder / f"{subject_id}_0000.nii.gz"
        if p.exists():
            return p
    raise FileNotFoundError(subject_id)


def predict_case(subject_id, side, fast=False, device="gpu"):
    out_path = prediction_path(METHOD, subject_id)
    if out_path.exists():
        return out_path

    roi = "femur_right" if side == "R" else "femur_left"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            str(Path(sys.prefix) / "bin" / "TotalSegmentator"),
            "-i", str(nifti_input(subject_id)),
            "-o", tmp,
            "--roi_subset", roi,
            "--device", device,
        ]
        if fast:
            cmd.append("--fast")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        pred = sitk.ReadImage(str(Path(tmp) / f"{roi}.nii.gz"))
        ref = sitk.ReadImage(str(ct_path(subject_id)))
        pred = sitk.Resample(pred, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        sitk.WriteImage(pred > 0, str(out_path), True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator femur predictions")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--fast", action="store_true", help="use the 3 mm model")
    parser.add_argument("--device", default="gpu", help="gpu or cpu")
    args = parser.parse_args()

    sides = load_sides()
    cases = load_splits()[args.split]
    for i, cid in enumerate(cases, 1):
        path = predict_case(cid, sides[cid], fast=args.fast, device=args.device)
        print(f"[{i}/{len(cases)}] {cid} -> {path}")


if __name__ == "__main__":
    main()
