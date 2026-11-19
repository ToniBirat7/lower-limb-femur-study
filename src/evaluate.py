"""
Score saved predictions against the reference masks with one shared code path.

Per case: Dice, HD95 (mm), average symmetric surface distance (mm).
Summary: mean, SD, bootstrap 95% CI of the mean; paired Wilcoxon tests of
each method against the best one.

    python src/evaluate.py --split test --methods classical monai nnunet totalseg
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
from scipy.stats import wilcoxon

from src.config import RESULTS_DIR, load_splits, mask_path, prediction_path
from src.metrics import evaluate_case

METRICS = ["Dice", "HD95_mm", "ASD_mm"]


def bootstrap_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def score_method(method, cases):
    rows = []
    for cid in cases:
        pred_file = prediction_path(method, cid)
        if not pred_file.exists():
            continue
        gt = sitk.ReadImage(str(mask_path(cid)))
        pred = sitk.ReadImage(str(pred_file))
        if pred.GetSize() != gt.GetSize():
            pred = sitk.Resample(pred, gt, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        rows.append({"subject_id": cid, **evaluate_case(pred > 0, gt > 0)})
    return pd.DataFrame(rows)


def summarise(per_method):
    lines = []
    for method, df in per_method.items():
        row = {"method": method, "n": len(df)}
        for m in METRICS:
            lo, hi = bootstrap_ci(df[m].dropna())
            row[m] = f"{df[m].mean():.3f} +/- {df[m].std():.3f} [{lo:.3f}, {hi:.3f}]"
        lines.append(row)
    return pd.DataFrame(lines)


def paired_tests(per_method, reference):
    ref = per_method[reference].set_index("subject_id")
    rows = []
    for method, df in per_method.items():
        if method == reference:
            continue
        other = df.set_index("subject_id")
        common = ref.index.intersection(other.index)
        for m in METRICS:
            a, b = ref.loc[common, m], other.loc[common, m]
            p = wilcoxon(a, b).pvalue if len(common) > 5 and not np.allclose(a, b) else np.nan
            rows.append({"comparison": f"{reference} vs {method}", "metric": m, "n": len(common),
                         "median_difference": round(float(np.median(a - b)), 4), "p_value": p})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--methods", nargs="+", default=["classical", "monai", "nnunet", "totalseg"])
    args = parser.parse_args()

    cases = load_splits()[args.split]
    RESULTS_DIR.mkdir(exist_ok=True)
    per_method = {}
    for method in args.methods:
        df = score_method(method, cases)
        if df.empty:
            print(f"{method}: no predictions for {args.split}")
            continue
        df.to_csv(RESULTS_DIR / f"segmentation_{method}_{args.split}.csv", index=False)
        per_method[method] = df
        print(f"{method}: {len(df)} cases, Dice {df.Dice.mean():.4f}")

    summary = summarise(per_method)
    summary.to_csv(RESULTS_DIR / f"segmentation_summary_{args.split}.csv", index=False)
    print("\n" + summary.to_string(index=False))

    if len(per_method) > 1:
        best = max(per_method, key=lambda k: per_method[k].Dice.mean())
        tests = paired_tests(per_method, best)
        tests.to_csv(RESULTS_DIR / f"segmentation_tests_{args.split}.csv", index=False)
        print("\n" + tests.to_string(index=False))


if __name__ == "__main__":
    main()
