"""Paths shared by all scripts. Set FEMUR_DATA to move the data root."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FEMUR_DATA", REPO_ROOT / "data"))

HFVALID_DIR = DATA_DIR / "HFValid"
SUBJECTS_DIR = HFVALID_DIR / "Subjects"
MASKS_DIR = HFVALID_DIR / "masks"
METADATA_CSV = HFVALID_DIR / "HFValid_Metadata.csv"
NNUNET_RAW = DATA_DIR / "nnunet" / "nnUNet_raw" / "Dataset001_Femur"
PREDICTIONS_DIR = DATA_DIR / "predictions"
WEIGHTS_DIR = DATA_DIR / "weights"

SPLITS_JSON = REPO_ROOT / "splits.json"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


def ct_path(subject_id):
    return SUBJECTS_DIR / subject_id / f"{subject_id}.nrrd"


def stl_path(subject_id):
    return SUBJECTS_DIR / subject_id / f"{subject_id}.stl"


def landmarks_path(subject_id):
    return SUBJECTS_DIR / subject_id / f"{subject_id}_palp.inp"


def mask_path(subject_id):
    return MASKS_DIR / f"{subject_id}_mask.nii.gz"


def prediction_path(method, subject_id):
    return PREDICTIONS_DIR / method / f"{subject_id}.nii.gz"


def load_splits():
    import json

    with open(SPLITS_JSON) as f:
        return json.load(f)


def load_sides():
    """Subject ID -> 'R' or 'L' (the scanned femur)."""
    import csv

    with open(METADATA_CSV, newline="") as f:
        return {row["PatientID"]: row["Side"] for row in csv.DictReader(f, delimiter=";")}
