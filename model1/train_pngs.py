"""
train_pngs.py — Training script for the temporal 3D U-Net on full mammogram PNGs.

Differences from train.py (patches):
- Uses MammogramPNGDataset from dataset3d_pngs.py
- Input size 512x512 instead of 256x256
- Batch size 4 (each sample is 4x larger than patches)
- Paths point to dataset_processed (cancerous/cancer_free split)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from experiments.classification.whole_mamms.dataset3d_pngs import MammogramPNGDataset
from model1.unet3d_model.unet3d import UNet3D

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CANCEROUS_TRAIN   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancerous"
CANCER_FREE_TRAIN = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancer_free"
CANCEROUS_VAL     = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val\cancerous"
CANCER_FREE_VAL   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val\cancer_free"

EPOCHS      = 50
BATCH_SIZE  = 8
NUM_WORKERS = 12
LR          = 1e-5
CHECKPOINT  = "best_model_pngs.pth"

# -------------------------------------------------------------------------
# Loss functions
# -------------------------------------------------------------------------

def dice_loss(pred, target, pad_mask):
    pred = torch.sigmoid(pred)
    B, C, T, H, W = pred.shape

    loss  = torch.tensor(0.0, device=pred.device)
    count = 0

    for t in range(T):
        valid = pad_mask[:, t]
        if valid.sum() == 0:
            continue

        p = pred[valid, :, t, :, :].reshape(valid.sum(), -1)
        g = target[valid, :, t, :, :].reshape(valid.sum(), -1)

        intersection = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)
        # Smooth denominator prevents degenerate all-zero solution
        dice = (2 * intersection + 1e-6) / (denom + 1e-6)
        loss += (1 - dice.mean())
        count += 1

    return loss / count if count > 0 else loss


def bce_loss(pred, target, pad_mask, criterion):
    """
    BCE loss over non-padded positions only.

    pred:     [B, 1, T, H, W]
    target:   [B, 1, T, H, W]
    pad_mask: [B, T]
    """
    mask = pad_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B,1,T,1,1]
    mask = mask.expand_as(pred)

    pred_valid   = pred[mask]
    target_valid = target[mask]

    if pred_valid.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return criterion(pred_valid, target_valid)


def dice_score(pred, target, pad_mask):
    """
    Hard Dice score for logging.
    Only computed on non-padded slices that have tumor in ground truth.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    B, C, T, H, W = pred.shape
    scores = []

    for t in range(T):
        valid = pad_mask[:, t]
        if valid.sum() == 0:
            continue

        p = pred[valid, :, t, :, :].reshape(valid.sum(), -1)
        g = target[valid, :, t, :, :].reshape(valid.sum(), -1)

        has_tumor = g.sum(dim=1) > 0
        if has_tumor.sum() == 0:
            continue

        p = p[has_tumor]
        g = g[has_tumor]

        intersection = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)
        dice  = (2 * intersection) / (denom + 1e-8)
        scores.append(dice.mean())

    if len(scores) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(scores).mean()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Datasets ---
    train_dataset = MammogramPNGDataset(
        cancerous_dir=CANCEROUS_TRAIN,
        cancer_free_dir=CANCER_FREE_TRAIN,
    )
    val_dataset = MammogramPNGDataset(
        cancerous_dir=CANCEROUS_VAL,
        cancer_free_dir=CANCER_FREE_VAL,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # --- Imbalance: WeightedRandomSampler ---
    n_pos = sum(1 for s in train_dataset.samples if s['label'] == 1)
    n_neg = sum(1 for s in train_dataset.samples if s['label'] == 0)
    print(f"Train positives: {n_pos}  negatives: {n_neg}  ratio: {n_neg/max(n_pos,1):.1f}:1")

    weights = [n_neg / n_pos if s['label'] == 1 else 1.0
               for s in train_dataset.samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Batches per epoch: {len(train_loader)}")

    # --- Model ---
    model = UNet3D(in_channels=1, num_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    scaler = GradScaler("cuda")

    best_val_dice = -1.0

    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        # ---- Training ----
        model.train()
        train_loss = 0.0

        for i, (imgs, masks, pad_mask, label) in enumerate(train_loader):

            if i == 0:
                print(f"  imgs shape:     {imgs.shape}")
                print(f"  masks shape:    {masks.shape}")
                print(f"  pad_mask shape: {pad_mask.shape}")
                print(f"  mask sum:       {masks.sum().item():.0f}")

            imgs     = imgs.to(device)
            masks    = masks.to(device)
            pad_mask = pad_mask.to(device)

            optimizer.zero_grad()

            with autocast("cuda"):
                preds = model(imgs)
                b     = bce_loss(preds, masks, pad_mask, criterion)
                d     = dice_loss(preds, masks, pad_mask)
                loss  = b + d

            if torch.isnan(loss):
                print(f"  [WARNING] NaN loss at batch {i}, skipping")
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if i % 100 == 0:
                print(f"  Batch {i}/{len(train_loader)}  loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        print(f"Train loss: {train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for imgs, masks, pad_mask, label in val_loader:

                imgs     = imgs.to(device)
                masks    = masks.to(device)
                pad_mask = pad_mask.to(device)

                with autocast("cuda"):
                    preds = model(imgs)

                b = bce_loss(preds, masks, pad_mask, criterion)
                d = dice_loss(preds, masks, pad_mask)
                val_loss += (b + d).item()
                val_dice += dice_score(preds, masks, pad_mask).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        print(f"Val loss:  {val_loss:.4f}")
        print(f"Val Dice:  {val_dice:.4f}")

        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Best model saved (Dice: {best_val_dice:.4f})")


if __name__ == "__main__":
    main()