"""
Header audit of all HFValid scans: size, spacing, origin, direction, HU range,
missing files and landmark coordinates.

    python src/audit.py [DATA_DIR] [OUT_CSV]
"""

import os
import glob
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tabulate import tabulate

DATA = os.environ.get("FEMUR_DATA", "data")


def parse_palp_inp(inp_path):
    """
    Parses PatXXX_palp.inp key-value landmark file.
    Returns dictionary of landmark_name -> (x, y, z).
    Key landmarks: femurHead, headDir, greatTroch, kneeCenter.
    """
    raw_vals = {}
    if not os.path.exists(inp_path):
        return raw_vals
    
    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                parts = [p.strip() for p in line.split("=", 1)]
                if len(parts) == 2:
                    try:
                        raw_vals[parts[0]] = float(parts[1])
                    except ValueError:
                        pass

    grouped = {}
    for k, v in raw_vals.items():
        if k.endswith(("X", "Y", "Z")):
            prefix = k[:-1]
            axis = k[-1]
            if prefix not in grouped:
                grouped[prefix] = {}
            grouped[prefix][axis] = v

    landmarks = {}
    for prefix, coords in grouped.items():
        if "X" in coords and "Y" in coords and "Z" in coords:
            landmarks[prefix] = (coords["X"], coords["Y"], coords["Z"])
            
    return landmarks


def audit_hfvalid(data_dir, output_csv="results/data/hfvalid_audit.csv"):
    data_path = Path(data_dir)
    subject_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("Pat")])
    if not subject_dirs:
        nested = data_path / "Subjects"
        if nested.exists():
            subject_dirs = sorted([d for d in nested.iterdir() if d.is_dir() and d.name.startswith("Pat")])

    print(f"Found {len(subject_dirs)} subject folders in {data_path}.")
    records = []

    for sdir in subject_dirs:
        pat_id = sdir.name
        nrrd_file = sdir / f"{pat_id}.nrrd"
        stl_file = sdir / f"{pat_id}.stl"
        palp_file = sdir / f"{pat_id}_palp.inp"

        has_nrrd = nrrd_file.exists()
        has_stl = stl_file.exists()
        has_palp = palp_file.exists()

        record = {
            "Patient_ID": pat_id,
            "Has_NRRD": has_nrrd,
            "Has_STL": has_stl,
            "Has_Palp": has_palp,
            "Size_X": None,
            "Size_Y": None,
            "Size_Z": None,
            "Spacing_X": None,
            "Spacing_Y": None,
            "Spacing_Z": None,
            "Origin_X": None,
            "Origin_Y": None,
            "Origin_Z": None,
            "Min_HU": None,
            "Max_HU": None,
            "Mean_HU": None,
            "FemurHead_X": None,
            "FemurHead_Y": None,
            "FemurHead_Z": None,
            "KneeCenter_X": None,
            "KneeCenter_Y": None,
            "KneeCenter_Z": None,
        }

        if has_palp:
            lm = parse_palp_inp(palp_file)
            if "femurHead" in lm:
                record["FemurHead_X"] = round(lm["femurHead"][0], 2)
                record["FemurHead_Y"] = round(lm["femurHead"][1], 2)
                record["FemurHead_Z"] = round(lm["femurHead"][2], 2)
            if "kneeCenter" in lm:
                record["KneeCenter_X"] = round(lm["kneeCenter"][0], 2)
                record["KneeCenter_Y"] = round(lm["kneeCenter"][1], 2)
                record["KneeCenter_Z"] = round(lm["kneeCenter"][2], 2)

        if has_nrrd:
            try:
                img = sitk.ReadImage(str(nrrd_file))
                size = img.GetSize()
                spacing = img.GetSpacing()
                origin = img.GetOrigin()

                record["Size_X"] = size[0]
                record["Size_Y"] = size[1]
                record["Size_Z"] = size[2]

                record["Spacing_X"] = round(spacing[0], 4)
                record["Spacing_Y"] = round(spacing[1], 4)
                record["Spacing_Z"] = round(spacing[2], 4)

                record["Origin_X"] = round(origin[0], 2)
                record["Origin_Y"] = round(origin[1], 2)
                record["Origin_Z"] = round(origin[2], 2)

                arr = sitk.GetArrayViewFromImage(img)
                record["Min_HU"] = float(np.min(arr))
                record["Max_HU"] = float(np.max(arr))
                record["Mean_HU"] = round(float(np.mean(arr)), 2)
            except Exception as e:
                record["Notes"] = f"Error reading NRRD: {e}"

        records.append(record)

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows to {output_csv}")
    print("\nSummary Statistics:")
    print(tabulate(df.describe().T, headers="keys", tablefmt="github"))
    return df


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "HFValid")
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/data/hfvalid_audit.csv"
    audit_hfvalid(data_dir, output_path)
