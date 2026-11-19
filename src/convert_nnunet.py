"""
Write the nnU-Net v2 raw dataset (Dataset001_Femur): train and validation cases in
imagesTr/labelsTr, test cases in imagesTs/labelsTs.

    python src/convert_nnunet.py
"""

import json
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import SimpleITK as sitk
from tqdm import tqdm

DATA = os.environ.get("FEMUR_DATA", "data")


def convert_single_case(task):
    pat_id, nrrd_path, mask_path, out_img_path, out_lbl_path, overwrite = task
    if out_img_path.exists() and out_lbl_path.exists() and not overwrite:
        return pat_id, "skipped"

    try:
        # Convert NRRD to NIfTI format with _0000 identifier
        img = sitk.ReadImage(str(nrrd_path))
        sitk.WriteImage(img, str(out_img_path), True)

        # Copy/convert mask
        mask = sitk.ReadImage(str(mask_path))
        sitk.WriteImage(mask, str(out_lbl_path), True)

        return pat_id, "success"
    except Exception as e:
        return pat_id, f"error: {e}"


def convert_dataset_to_nnunet(
    data_dir=os.path.join(DATA, "HFValid"),
    masks_dir=os.path.join(DATA, "HFValid/masks"),
    splits_json="splits.json",
    nnunet_raw_dir=os.path.join(DATA, "nnunet/nnUNet_raw"),
    dataset_name="Dataset001_Femur",
    num_workers=4,
    overwrite=False
):
    data_path = Path(data_dir)
    subj_dir = data_path / "Subjects" if (data_path / "Subjects").exists() else data_path
    masks_path = Path(masks_dir)

    with open(splits_json, "r") as f:
        splits = json.load(f)

    train_ids = set(splits["train"] + splits["val"])
    test_ids = set(splits["test"])

    out_dataset_dir = Path(nnunet_raw_dir) / dataset_name
    imagesTr = out_dataset_dir / "imagesTr"
    labelsTr = out_dataset_dir / "labelsTr"
    imagesTs = out_dataset_dir / "imagesTs"
    labelsTs = out_dataset_dir / "labelsTs"

    for d in [imagesTr, labelsTr, imagesTs, labelsTs]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Preparing nnU-Net dataset at {out_dataset_dir}")
    print(f"  Training + Val cases: {len(train_ids)}")
    print(f"  Testing cases:        {len(test_ids)}")

    tasks = []
    for pat_id in sorted(list(train_ids)):
        nrrd_path = subj_dir / pat_id / f"{pat_id}.nrrd"
        mask_path = masks_path / f"{pat_id}_mask.nii.gz"
        out_img = imagesTr / f"{pat_id}_0000.nii.gz"
        out_lbl = labelsTr / f"{pat_id}.nii.gz"
        tasks.append((pat_id, nrrd_path, mask_path, out_img, out_lbl, overwrite))

    for pat_id in sorted(list(test_ids)):
        nrrd_path = subj_dir / pat_id / f"{pat_id}.nrrd"
        mask_path = masks_path / f"{pat_id}_mask.nii.gz"
        out_img = imagesTs / f"{pat_id}_0000.nii.gz"
        out_lbl = labelsTs / f"{pat_id}.nii.gz"
        tasks.append((pat_id, nrrd_path, mask_path, out_img, out_lbl, overwrite))

    print(f"Converting {len(tasks)} volumes in parallel with {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for pat_id, status in tqdm(executor.map(convert_single_case, tasks), total=len(tasks)):
            if status.startswith("error"):
                print(f"  Error on {pat_id}: {status}")

    # Generate dataset.json
    dataset_json_data = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "femur": 1
        },
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz"
    }

    with open(out_dataset_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json_data, f, indent=2)

    print(f"\nWrote {out_dataset_dir}")
    print(f"Generated dataset.json with {len(train_ids)} training volumes.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert HFValid to nnUNet raw dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--workers", type=int, default=4, help="Worker processes")
    args = parser.parse_args()

    convert_dataset_to_nnunet(overwrite=args.overwrite, num_workers=args.workers)
