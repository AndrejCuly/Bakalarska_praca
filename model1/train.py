"""
train.py — Fixed training script for the temporal 3D U-Net on mammogram patches.

Key fixes over the previous version:
1. Loss is masked over padded time slices (pad_mask) so padding never
   contributes to gradients.
2. Imbalance handled ONCE via WeightedRandomSampler only — removed the
   redundant pos_weight in BCEWithLogitsLoss that was double-compensating.
3. Best model checkpoint now saves on best val Dice, not val loss.
   Val loss was diverging while Dice was the actual task metric.
4. Removed torch.roll augmentation — it wraps border pixels nonsensically.
5. Loss formula: equal weight BCE + Dice, no arbitrary /2 on BCE.
6. dataset3d.py now returns (images, masks, pad_mask, label) — train.py
   updated accordingly.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from dataset3d import Mammogram3DDataset
from unet3d_model.unet3d import UNet3D

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CANCER_TRAIN      = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer"
CANCER_FREE_TRAIN = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer_free"
CANCER_VAL        = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer"
CANCER_FREE_VAL   = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer_free"

EPOCHS      = 50
BATCH_SIZE  = 16
NUM_WORKERS = 8
LR          = 1e-4
CHECKPOINT  = "best_model.pth"

# -------------------------------------------------------------------------
# Loss functions
# -------------------------------------------------------------------------

def dice_loss(pred, target, pad_mask):
    """
    Soft Dice loss, computed only over non-padded time slices.

    pred:     [B, 1, T, H, W]  raw logits
    target:   [B, 1, T, H, W]  binary masks
    pad_mask: [B, T]            True = real slice, False = padding
    """
    pred = torch.sigmoid(pred)

    B, C, T, H, W = pred.shape

    loss = torch.tensor(0.0, device=pred.device)
    count = 0

    for t in range(T):
        # Only include samples where this time step is real (not padding)
        valid = pad_mask[:, t]          # [B] bool
        if valid.sum() == 0:
            continue

        p = pred[valid, :, t, :, :].reshape(valid.sum(), -1)
        g = target[valid, :, t, :, :].reshape(valid.sum(), -1)

        intersection = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)
        dice = (2 * intersection + 1e-6) / (denom + 1e-6)
        loss += (1 - dice.mean())
        count += 1

    return loss / count if count > 0 else loss


def bce_loss(pred, target, pad_mask, criterion):
    """
    BCE loss, masked so padded slices are ignored.

    pred:     [B, 1, T, H, W]
    target:   [B, 1, T, H, W]
    pad_mask: [B, T]
    """
    B, C, T, H, W = pred.shape

    # Expand pad_mask to match pred shape: [B, 1, T, H, W]
    mask = pad_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, T, 1, 1]
    mask = mask.expand_as(pred)

    # Only compute loss on valid (non-padded) positions
    pred_valid   = pred[mask]
    target_valid = target[mask]

    if pred_valid.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return criterion(pred_valid, target_valid)


def dice_score(pred, target, pad_mask):
    """
    Hard Dice score for logging. Only over non-padded, non-empty mask slices.
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

        # Only count slices that actually have a tumor in the ground truth
        has_tumor = g.sum(dim=1) > 0
        if has_tumor.sum() == 0:
            continue

        p = p[has_tumor]
        g = g[has_tumor]

        intersection = (p * g).sum(dim=1)
        denom = p.sum(dim=1) + g.sum(dim=1)
        dice = (2 * intersection) / (denom + 1e-8)
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
    train_dataset = Mammogram3DDataset(
        cancer_dir=CANCER_TRAIN,
        cancer_free_dir=CANCER_FREE_TRAIN,
    )
    val_dataset = Mammogram3DDataset(
        cancer_dir=CANCER_VAL,
        cancer_free_dir=CANCER_FREE_VAL,
    )

    val_pos = sum(1 for s in val_dataset.samples if s['label'] == 1)
    val_neg = sum(1 for s in val_dataset.samples if s['label'] == 0)
    print(f"Val positives: {val_pos}  negatives: {val_neg}")

    # Count slices with actual tumor pixels
    import os
    from PIL import Image
    import numpy as np
    tumor_slices = 0
    for s in val_dataset.samples:
        if s['label'] != 1:
            continue
        for fname in s['filenames']:
            mask_path = os.path.join(s['masks_dir'], fname)
            if os.path.exists(mask_path):
                m = np.array(Image.open(mask_path).convert('L'))
                if m.max() > 0:
                    tumor_slices += 1
    print(f"Val tumor slices with actual white pixels: {tumor_slices}")


    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # --- Imbalance: WeightedRandomSampler (handled ONCE, not also in loss) ---
    # Separate positive and negative indices
    pos_indices = [i for i, s in enumerate(train_dataset.samples) if s['label'] == 1]
    neg_indices = [i for i, s in enumerate(train_dataset.samples) if s['label'] == 0]

    print(f"Train positives: {len(pos_indices)}  negatives: {len(neg_indices)}")

    # Weight so positives appear ~50% of the time in each batch
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
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
    # Small capacity by default — appropriate for ~150-200 positive patients
    # Increase level_channels if you have significantly more data
    model = UNet3D(in_channels=1, num_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # No pos_weight here — imbalance is handled by the sampler above
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Reduce LR if val Dice stops improving
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
                b = bce_loss(preds, masks, pad_mask, criterion)
                d = dice_loss(preds, masks, pad_mask)
                loss = b + d

            if torch.isnan(loss):
                print(f"  [WARNING] NaN loss at batch {i}, skipping")
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if i % 100 == 0:
                print(f"  Batch {i}/{len(train_loader)}  loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        print(f"Train loss: {train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss  = 0.0
        val_dice  = 0.0

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

        # Step scheduler on val Dice (higher = better)
        scheduler.step(val_dice)

        # Save best model on Dice, not loss
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Best model saved (Dice: {best_val_dice:.4f})")


if __name__ == "__main__":
    main()