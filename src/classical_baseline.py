"""
Classical femur segmentation baseline (no learning).

1. Bone threshold (200 to 2000 HU)
2. Largest 3D connected component
3. Morphological closing
4. Hole filling slice by slice in the axial plane, then in 3D
   (trabecular bone in elderly femurs is often below 200 HU, so the
   cortical shell has to be filled)

    python src/classical_baseline.py --split test
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

from src.config import ct_path, load_splits, prediction_path

METHOD = "classical"


def segment_femur_classical(ct, lower_hu=200, upper_hu=2000):
    bone = sitk.BinaryThreshold(ct, float(lower_hu), float(upper_hu), 1, 0)
    largest = sitk.RelabelComponent(sitk.ConnectedComponent(bone)) == 1
    closed = sitk.BinaryMorphologicalClosing(largest, [2, 2, 1])
    arr = sitk.GetArrayFromImage(closed).astype(bool)
    for z in range(arr.shape[0]):
        arr[z] = binary_fill_holes(arr[z])
    arr = binary_fill_holes(arr)
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(ct)
    return out


def main():
    parser = argparse.ArgumentParser(description="Classical threshold baseline")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    cases = load_splits()[args.split]
    for i, cid in enumerate(cases, 1):
        out = prediction_path(METHOD, cid)
        out.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(segment_femur_classical(sitk.ReadImage(str(ct_path(cid)))), str(out), True)
        print(f"[{i}/{len(cases)}] {cid}")


if __name__ == "__main__":
    main()
