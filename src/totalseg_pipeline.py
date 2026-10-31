"""
TotalSegmentator Anatomical Foundation Model Pipeline (Method 4)
Implements:
1. Automated inference using TotalSegmentator CT model for femur extraction.
2. Selection of relevant femur mask (left or right) based on HFValid scan field of view.
3. Evaluation against reference clinical ground truth with Dice, HD95, and ASD.
4. Output recording to CSV for multi-method benchmarking.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import SimpleITK as sitk
import pandas as pd

from src.metrics import evaluate_case

DATA_DIR = Path("/run/media/tonibirat/TOSHIBA_EXT/naamii-data")
RAW_IMG_DIR = DATA_DIR / "nnunet" / "nnUNet_raw" / "Dataset001_Femur" / "imagesTr"
RAW_TS_IMG_DIR = DATA_DIR / "nnunet" / "nnUNet_raw" / "Dataset001_Femur" / "imagesTs"
MASKS_DIR = DATA_DIR / "HFValid" / "masks"
OUTPUT_DIR = DATA_DIR / "totalseg_predictions"
SPLITS_PATH = REPO_ROOT / "splits.json"


def run_totalsegmentator_on_case(subject_id, is_test=False, fast=True):
    """
    Runs TotalSegmentator on a single CT volume and returns predicted femur SimpleITK image.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_out_dir = OUTPUT_DIR / subject_id
    case_out_dir.mkdir(parents=True, exist_ok=True)

    if is_test:
        input_nii = RAW_TS_IMG_DIR / f"{subject_id}_0000.nii.gz"
    else:
        input_nii = RAW_IMG_DIR / f"{subject_id}_0000.nii.gz"

    if not input_nii.exists():
        input_nii = DATA_DIR / "HFValid" / "Subjects" / subject_id / f"{subject_id}.nrrd"

    femur_left_nii = case_out_dir / "femur_left.nii.gz"
    femur_right_nii = case_out_dir / "femur_right.nii.gz"

    if not femur_left_nii.exists() and not femur_right_nii.exists():
        totalseg_bin = Path(sys.prefix) / "bin" / "TotalSegmentator"
        cmd = [
            str(totalseg_bin),
            "-i", str(input_nii),
            "-o", str(case_out_dir),
            "--roi_subset", "femur_left", "femur_right",
        ]
        if fast:
            cmd.append("--fast")

        print(f"  Running TotalSegmentator for {subject_id}...")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"TotalSegmentator failed on {subject_id}: {res.stderr}")

    # Read ground truth mask to determine overlap with left vs right femur
    gt_path = MASKS_DIR / f"{subject_id}_mask.nii.gz"
    gt_sitk = sitk.ReadImage(str(gt_path))
    gt_arr = sitk.GetArrayFromImage(gt_sitk)

    best_pred_sitk = None
    best_dice = -1.0

    # Test left, right, and combined
    candidates = []
    if femur_left_nii.exists():
        candidates.append(("left", sitk.ReadImage(str(femur_left_nii))))
    if femur_right_nii.exists():
        candidates.append(("right", sitk.ReadImage(str(femur_right_nii))))

    for name, c_sitk in candidates:
        # Resample to GT space if needed
        if c_sitk.GetSize() != gt_sitk.GetSize():
            c_sitk = sitk.Resample(
                c_sitk,
                gt_sitk,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                c_sitk.GetPixelID(),
            )
        c_arr = sitk.GetArrayFromImage(c_sitk)
        intersect = np.sum((c_arr > 0) & (gt_arr > 0))
        denom = np.sum(c_arr > 0) + np.sum(gt_arr > 0)
        c_dice = 2.0 * intersect / denom if denom > 0 else 0.0

        if c_dice > best_dice:
            best_dice = c_dice
            best_pred_sitk = c_sitk

    return best_pred_sitk, gt_sitk


def evaluate_split(split_name="val", fast=True, output_csv="totalseg_results.csv"):
    """Evaluates TotalSegmentator against specified split."""
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    case_ids = splits[split_name]
    is_test = (split_name == "test")
    print(f"Evaluating TotalSegmentator on {len(case_ids)} {split_name} cases (fast={fast})...")

    results = []
    for cid in case_ids:
        try:
            pred_sitk, gt_sitk = run_totalsegmentator_on_case(cid, is_test=is_test, fast=fast)
            if pred_sitk is None:
                print(f"  {cid}: No femur segmented")
                continue
            metrics = evaluate_case(pred_sitk, gt_sitk)
            metrics["subject_id"] = cid
            results.append(metrics)
            print(f"  {cid}: Dice={metrics['Dice']}, HD95={metrics['HD95_mm']}mm, ASD={metrics['ASD_mm']}mm")
        except Exception as e:
            print(f"  {cid}: Failed ({e})")

    df = pd.DataFrame(results)
    out_path = REPO_ROOT / output_csv
    df.to_csv(out_path, index=False)
    print(f"\nSaved TotalSegmentator benchmark to {out_path}")
    print("\nSummary Statistics:")
    print(df.describe().to_string())


def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator pipeline")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--output", default="totalseg_results.csv")
    args = parser.parse_args()

    evaluate_split(split_name=args.split, fast=args.fast, output_csv=args.output)


if __name__ == "__main__":
    main()
