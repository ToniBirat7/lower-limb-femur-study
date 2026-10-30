"""
Patient-Level Dataset Splitting for HFValid Femur Study
Generates a deterministic train / validation / test split with balanced laterality:
- 71 Train subjects
- 10 Validation subjects
- 20 Test subjects (held out, evaluated exactly once)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def generate_splits(audit_csv_path, output_json_path, seed=42):
    df = pd.read_csv(audit_csv_path)
    print(f"Loaded {len(df)} subjects from {audit_csv_path}")

    # Determine laterality from metadata or FemurHead_X coordinate
    # Positive FemurHead_X = Left femur; Negative FemurHead_X = Right femur
    df["Side"] = np.where(df["FemurHead_X"] > 0, "L", "R")
    
    np.random.seed(seed)
    
    left_pats = sorted(df[df["Side"] == "L"]["Patient_ID"].tolist())
    right_pats = sorted(df[df["Side"] == "R"]["Patient_ID"].tolist())

    np.random.shuffle(left_pats)
    np.random.shuffle(right_pats)

    # 101 subjects total: target 71 train (approx 70%), 10 val (approx 10%), 20 test (approx 20%)
    # Stratified by laterality:
    n_left = len(left_pats)
    n_right = len(right_pats)

    n_val_left = int(round(10 * n_left / 101))
    n_val_right = 10 - n_val_left

    n_test_left = int(round(20 * n_left / 101))
    n_test_right = 20 - n_test_left

    val_pats = sorted(left_pats[:n_val_left] + right_pats[:n_val_right])
    test_pats = sorted(left_pats[n_val_left:n_val_left + n_test_left] + right_pats[n_val_right:n_val_right + n_test_right])
    train_pats = sorted(left_pats[n_val_left + n_test_left:] + right_pats[n_val_right + n_test_right:])

    splits = {
        "metadata": {
            "seed": seed,
            "total_subjects": len(df),
            "train_count": len(train_pats),
            "val_count": len(val_pats),
            "test_count": len(test_pats),
        },
        "train": train_pats,
        "val": val_pats,
        "test": test_pats,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print("\nDataset split complete:")
    print(f"  Train: {len(train_pats)} subjects")
    print(f"  Val:   {len(val_pats)} subjects")
    print(f"  Test:  {len(test_pats)} subjects")
    print(f"Saved split specification to {output_json_path}")
    return splits


if __name__ == "__main__":
    import sys
    audit_csv = sys.argv[1] if len(sys.argv) > 1 else "/run/media/tonibirat/EA1AD7931AD75AE1/Projects/lower-limb-femur-study/hfvalid_audit.csv"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "/run/media/tonibirat/EA1AD7931AD75AE1/Projects/lower-limb-femur-study/splits.json"
    generate_splits(audit_csv, out_json)
