"""
Classical Segmentation Baseline for Femur CT
Pipeline:
1. Hounsfield Unit Thresholding (bone window: 200 to 1800 HU)
2. 3D Connected Component Analysis (isolate largest connected structure)
3. Binary Hole Filling & Morphological Closing
4. Quantitative Evaluation (Dice, HD95 in mm, ASD in mm)
"""

import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from metrics import evaluate_case


def segment_femur_classical(ct_sitk, lower_hu=200, upper_hu=1800):
    """
    Classical thresholding + largest connected component baseline.
    """
    # 1. Thresholding
    thresh_img = sitk.BinaryThreshold(
        ct_sitk,
        lowerThreshold=float(lower_hu),
        upperThreshold=float(upper_hu),
        insideValue=1,
        outsideValue=0
    )

    # 2. 3D Connected Components
    cc_filter = sitk.ConnectedComponentImageFilter()
    labeled = cc_filter.Execute(thresh_img)

    # Find label with largest size
    relabeled = sitk.RelabelComponent(labeled)
    # Label 1 is guaranteed to be the largest component
    largest_cc = (relabeled == 1)

    # 3. Morphological closing to seal cortical holes
    closed_mask = sitk.BinaryMorphologicalClosing(largest_cc, [2, 2, 2])
    filled_mask = sitk.BinaryFillhole(closed_mask)

    return filled_mask


def evaluate_baseline_on_split(
    split_name="val",
    data_dir="/run/media/tonibirat/TOSHIBA_EXT/naamii-data/HFValid",
    masks_dir="/run/media/tonibirat/TOSHIBA_EXT/naamii-data/HFValid/masks",
    splits_json="/run/media/tonibirat/EA1AD7931AD75AE1/Projects/lower-limb-femur-study/splits.json",
    out_csv="classical_baseline_results.csv"
):
    data_path = Path(data_dir)
    subj_dir = data_path / "Subjects" if (data_path / "Subjects").exists() else data_path
    masks_path = Path(masks_dir)

    with open(splits_json, "r") as f:
        splits = json.load(f)

    target_ids = splits.get(split_name, [])
    print(f"Evaluating classical baseline on {len(target_ids)} subjects in '{split_name}' split...")

    results = []
    for pat_id in tqdm(target_ids):
        ct_path = subj_dir / pat_id / f"{pat_id}.nrrd"
        mask_path = masks_path / f"{pat_id}_mask.nii.gz"

        if not ct_path.exists() or not mask_path.exists():
            continue

        ct_img = sitk.ReadImage(str(ct_path))
        gt_mask = sitk.ReadImage(str(mask_path))

        pred_mask = segment_femur_classical(ct_img)
        metrics = evaluate_case(pred_mask, gt_mask)
        metrics["Patient_ID"] = pat_id
        results.append(metrics)

    df = pd.DataFrame(results)
    cols = ["Patient_ID", "Dice", "HD95_mm", "ASD_mm"]
    df = df[cols]
    df.to_csv(out_csv, index=False)

    print(f"\nClassical Baseline Evaluation Complete ({split_name}):")
    print(tabulate(df, headers="keys", tablefmt="github"))

    print("\nSummary Statistics:")
    summary = df[["Dice", "HD95_mm", "ASD_mm"]].describe().T
    print(tabulate(summary, headers="keys", tablefmt="github"))
    return df


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    evaluate_baseline_on_split(split)
