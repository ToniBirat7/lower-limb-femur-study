"""
Agreement of planning measurements taken on predicted masks with the same
measurements on the reference meshes (Bland-Altman bias and 95% limits of
agreement, mean absolute difference).

Run planning.py first for the reference and each method on the same split.

    python src/planning_agreement.py --split test --methods nnunet totalseg
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.config import result_path

MEASURES = ["head_diameter_mm", "nsa_deg", "anteversion_deg", "ama_deg"]


def agreement(ref, other):
    merged = ref.merge(other, on="subject_id", suffixes=("_ref", "_pred"))
    rows = []
    for m in MEASURES:
        d = (merged[f"{m}_pred"] - merged[f"{m}_ref"]).dropna()
        rows.append({
            "measure": m,
            "n": len(d),
            "bias": round(d.mean(), 2),
            "loa_low": round(d.mean() - 1.96 * d.std(), 2),
            "loa_high": round(d.mean() + 1.96 * d.std(), 2),
            "mean_abs_diff": round(d.abs().mean(), 2),
            "max_abs_diff": round(d.abs().max(), 2),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Planning agreement with the reference")
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", nargs="+", default=["nnunet"])
    args = parser.parse_args()

    ref = pd.read_csv(result_path("planning", f"reference_{args.split}.csv"))
    tables = []
    for method in args.methods:
        df = agreement(ref, pd.read_csv(result_path("planning", f"{method}_{args.split}.csv")))
        df.insert(0, "method", method)
        tables.append(df)
    out = pd.concat(tables)
    out.to_csv(result_path("planning", f"agreement_{args.split}.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
