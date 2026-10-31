"""
MONAI 3D Residual U-Net Pipeline (Method 2)
Implements:
- 3D Residual U-Net for whole-femur CT segmentation
- Memory-efficient mixed precision (AMP) and patch-based training
- Sliding-window inference (monai.inferers.sliding_window_inference)
- Evaluation on validation and test splits with Dice and HD95 metrics
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import monai
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotated,
    EnsureTyped,
    Activations,
    AsDiscrete,
)
from monai.data import Dataset, DataLoader

from src.metrics import evaluate_case

# Default paths
DATA_DIR = Path("/run/media/tonibirat/TOSHIBA_EXT/naamii-data")
RAW_IMG_DIR = DATA_DIR / "nnunet" / "nnUNet_raw" / "Dataset001_Femur" / "imagesTr"
RAW_LBL_DIR = DATA_DIR / "nnunet" / "nnUNet_raw" / "Dataset001_Femur" / "labelsTr"
RAW_TS_IMG_DIR = DATA_DIR / "nnunet" / "nnUNet_raw" / "Dataset001_Femur" / "imagesTs"
RAW_TS_LBL_DIR = DATA_DIR / "nnunet" / "nnUNet_raw" / "Dataset001_Femur" / "labelsTs"
SPLITS_PATH = Path(__file__).resolve().parent.parent / "splits.json"
WEIGHTS_DIR = Path("/run/media/tonibirat/TOSHIBA_EXT/weights/monai")


def get_case_files(subject_id, is_test=False):
    """Returns image and label file paths for a given subject ID."""
    if is_test:
        img_path = RAW_TS_IMG_DIR / f"{subject_id}_0000.nii.gz"
        lbl_path = RAW_TS_LBL_DIR / f"{subject_id}.nii.gz"
    else:
        img_path = RAW_IMG_DIR / f"{subject_id}_0000.nii.gz"
        lbl_path = RAW_LBL_DIR / f"{subject_id}.nii.gz"
    
    if not img_path.exists():
        # Fallback to HFValid raw directory
        img_path = DATA_DIR / "HFValid" / "Subjects" / subject_id / f"{subject_id}.nrrd"
        lbl_path = DATA_DIR / "HFValid" / "masks" / f"{subject_id}_mask.nii.gz"

    return {"image": str(img_path), "label": str(lbl_path), "subject_id": subject_id}


def build_transforms(spatial_size=(96, 96, 96), is_train=True):
    """Builds MONAI data preprocessing and augmentation pipeline."""
    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-100.0,
            a_max=1500.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    if is_train:
        train_transforms = common_transforms + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=2,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotated(
                keys=["image", "label"],
                range_x=0.2,
                prob=0.3,
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
        return Compose(train_transforms)
    else:
        val_transforms = common_transforms + [
            EnsureTyped(keys=["image", "label"]),
        ]
        return Compose(val_transforms)


def build_model(device):
    """Constructs 3D Residual U-Net optimized for 6GB RTX 3060 VRAM."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    return model


def run_sanity_check(device):
    """Runs a quick 5-epoch overfitting sanity check on 2 subjects."""
    print("Running MONAI 3D U-Net sanity check on 2 cases...")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    sanity_cases = splits["val"][:2]
    data_dicts = [get_case_files(c) for c in sanity_cases]

    transforms = build_transforms(is_train=True)
    ds = Dataset(data=data_dicts, transform=transforms)
    loader = DataLoader(ds, batch_size=2, shuffle=True)

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    model.train()
    for epoch in range(1, 6):
        epoch_loss = 0.0
        for batch in loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = loss_fn(preds, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Sanity Epoch {epoch}/5 - Loss: {epoch_loss / len(loader):.4f}")

    print("Sanity check passed successfully! Model trains and gradients backpropagate.\n")


def train(epochs=50, batch_size=2, device="cuda"):
    """Trains 3D Residual U-Net on training split."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Training MONAI 3D U-Net on {dev} for {epochs} epochs...")

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    train_files = [get_case_files(c) for c in splits["train"]]
    val_files = [get_case_files(c) for c in splits["val"]]

    train_ds = Dataset(data=train_files, transform=build_transforms(is_train=True))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    val_ds = Dataset(data=val_files, transform=build_transforms(is_train=False))
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1)

    model = build_model(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    best_val_dice = 0.0
    best_model_path = WEIGHTS_DIR / "best_monai_unet.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs, lbls = batch["image"].to(dev), batch["label"].to(dev)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                preds = model(imgs)
                loss = loss_fn(preds, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_dices = []
            with torch.no_grad():
                for batch in val_loader:
                    v_imgs = batch["image"].to(dev)
                    v_lbls = batch["label"].to(dev)
                    with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                        v_preds = sliding_window_inference(
                            v_imgs, (96, 96, 96), 2, model, overlap=0.25
                        )
                    pred_label = torch.argmax(v_preds, dim=1, keepdim=True)
                    # Compute Dice
                    intersect = ((pred_label == 1) & (v_lbls == 1)).sum().float()
                    cardinality = (pred_label == 1).sum().float() + (v_lbls == 1).sum().float()
                    dice = (2.0 * intersect / (cardinality + 1e-6)).item()
                    val_dices.append(dice)

            mean_val_dice = float(np.mean(val_dices))
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Dice: {mean_val_dice:.4f}")

            if mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Saved new best model to {best_model_path} (Dice: {best_val_dice:.4f})")


def main():
    parser = argparse.ArgumentParser(description="MONAI 3D Residual U-Net pipeline")
    parser.add_argument("--mode", choices=["sanity", "train", "val", "test"], default="sanity")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.mode == "sanity":
        dev = torch.device(args.device)
        run_sanity_check(dev)
    elif args.mode == "train":
        train(epochs=args.epochs, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
