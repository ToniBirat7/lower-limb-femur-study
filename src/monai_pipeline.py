"""
MONAI 3D U-Net (residual units) for femur segmentation.

Images are reoriented to RAS, resampled to 1.5 mm, clipped to [-200, 1500] HU
and scaled to [0, 1]. Training uses 96^3 patches on a 6 GB GPU with bfloat16
mixed precision (float16 overflowed to NaN on some patches); inference uses
sliding windows and maps the prediction back to the original CT grid.

    python src/monai_pipeline.py --mode sanity            # overfit two cases
    python src/monai_pipeline.py --mode train --hours 2 --epochs 1000
    python src/monai_pipeline.py --mode predict --split test
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism

from src.config import NNUNET_RAW, WEIGHTS_DIR, load_splits, mask_path, prediction_path, result_path

METHOD = "monai"
PATCH = (96, 96, 96)
CKPT = WEIGHTS_DIR / "monai" / "unet_best.pt"


def case(cid, split):
    folder = "imagesTs" if split == "test" else "imagesTr"
    return {"image": str(NNUNET_RAW / folder / f"{cid}_0000.nii.gz"), "label": str(mask_path(cid)), "id": cid}


def base_transforms(keys):
    return [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys=["image"], a_min=-200.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True),
    ]


def train_transforms():
    return Compose(base_transforms(["image", "label"]) + [
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=PATCH,
                               pos=2, neg=1, num_samples=2, image_key="image", image_threshold=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.05, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ])


def eval_transforms():
    return Compose(base_transforms(["image", "label"]) + [EnsureTyped(keys=["image", "label"])])


def build_model(device):
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
        num_res_units=2, norm=Norm.INSTANCE,
    ).to(device)


def dice_of(pred, label):
    p, g = pred.bool(), label.bool()
    return (2.0 * (p & g).sum() / (p.sum() + g.sum() + 1e-6)).item()


def validate(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for batch in loader:
            logits = sliding_window_inference(batch["image"].to(device), PATCH, 4, model, overlap=0.25)
            scores.append(dice_of(logits.argmax(1, keepdim=True), batch["label"].to(device)))
    return float(np.mean(scores))


def train(hours, max_epochs, device, cases_train, cases_val, val_every=10):
    set_determinism(seed=0)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    train_ds = CacheDataset([case(c, "train") for c in cases_train], train_transforms(), cache_rate=1.0, num_workers=6)
    val_ds = CacheDataset([case(c, "val") for c in cases_val], eval_transforms(), cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    log = []
    best, start = -1.0, time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            x, y = batch["image"].to(device), batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step()
            total += loss.item()
        sched.step()
        row = {"epoch": epoch, "train_loss": total / len(train_loader), "minutes": (time.time() - start) / 60}
        out_of_time = (time.time() - start) / 3600 > hours
        if epoch % val_every == 0 or epoch == max_epochs or out_of_time:
            row["val_dice"] = validate(model, val_loader, device)
            if row["val_dice"] > best:
                best = row["val_dice"]
                torch.save(model.state_dict(), CKPT)
        log.append(row)
        print(", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if out_of_time:
            print(f"time budget of {hours} h reached")
            break
    import pandas as pd

    pd.DataFrame(log).to_csv(result_path("training", "monai_log.csv"), index=False)
    print(f"best validation Dice {best:.4f}, checkpoint {CKPT}")


def predict(split, device):
    pre = Compose(base_transforms(["image"]) + [EnsureTyped(keys=["image"])])
    post = Compose([
        Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False, to_tensor=True),
        AsDiscreted(keys="pred", argmax=True),
    ])
    cases = load_splits()[split]
    ds = Dataset([{"image": case(c, split)["image"], "id": c} for c in cases], pre)
    loader = DataLoader(ds, batch_size=1, num_workers=2)

    model = build_model(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch["pred"] = sliding_window_inference(batch["image"].to(device), PATCH, 4, model, overlap=0.5)
            for item in decollate_batch(batch):
                item["pred"] = item["pred"].float().cpu()
                item = post(item)
                pred = item["pred"][0].numpy().astype(np.uint8)
                affine = item["pred"].meta["affine"].numpy()
                out = prediction_path(METHOD, item["id"])
                out.parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(pred, affine), str(out))
                print(f"  {item['id']} -> {out}", flush=True)


def sanity(device):
    """Overfit two validation cases: the loss must fall well below its starting value."""
    ids = load_splits()["val"][:2]
    ds = CacheDataset([case(c, "val") for c in ids], train_transforms(), cache_rate=1.0)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    model = build_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    for epoch in range(1, 41):
        for batch in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch["image"].to(device)), batch["label"].to(device))
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch}: loss {loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="MONAI 3D U-Net")
    parser.add_argument("--mode", choices=["sanity", "train", "predict"], default="sanity")
    parser.add_argument("--hours", type=float, default=2.0, help="training time budget")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    if args.mode == "sanity":
        sanity(device)
    elif args.mode == "train":
        train(args.hours, args.epochs, device, splits["train"], splits["val"])
    else:
        predict(args.split, device)


if __name__ == "__main__":
    main()
