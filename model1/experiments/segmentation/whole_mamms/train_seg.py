"""
train_seg.py

Segmentation training for Model 1 (3D U-Net) on whole mammograms.
This experiment is expected to fail due to extreme class imbalance
between tumor pixels and background — reported as a failed experiment.

Place in: model1/experiments/segmentation/whole_mamms/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

from model1.experiments.classification.whole_mamms.dataset3d_pngs import MammogramPNGDataset
from model1.unet3d_model.unet3d import UNet3D

# -------------------------------------------------------------------------
CANCEROUS_TRAIN   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancerous"
CANCER_FREE_TRAIN = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancer_free"
CANCEROUS_VAL     = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val\cancerous"
CANCER_FREE_VAL   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val\cancer_free"

EPOCHS      = 10
BATCH_SIZE  = 4
NUM_WORKERS = 8
LR          = 1e-4
CHECKPOINT  = os.path.join(os.path.dirname(__file__), 'results', 'best_model_seg.pth')
# -------------------------------------------------------------------------


def dice_loss(pred, target, pad_mask):
    pred  = torch.sigmoid(pred)
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
        dice  = (2 * intersection + 1e-6) / (denom + 1e-6)
        loss += (1 - dice.mean())
        count += 1
    return loss / count if count > 0 else loss


def bce_loss(pred, target, pad_mask, criterion):
    mask = pad_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(pred)
    pred_valid   = pred[mask]
    target_valid = target[mask]
    if pred_valid.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return criterion(pred_valid, target_valid)


def dice_score(pred, target, pad_mask):
    pred  = (torch.sigmoid(pred) > 0.5).float()
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
        scores.append(((2 * intersection) / (denom + 1e-8)).mean())
    return torch.stack(scores).mean() if scores else torch.tensor(0.0)


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = MammogramPNGDataset(
        cancerous_dir=CANCEROUS_TRAIN, cancer_free_dir=CANCER_FREE_TRAIN)
    val_dataset   = MammogramPNGDataset(
        cancerous_dir=CANCEROUS_VAL,   cancer_free_dir=CANCER_FREE_VAL)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    n_pos = sum(1 for s in train_dataset.samples if s['label'] == 1)
    n_neg = sum(1 for s in train_dataset.samples if s['label'] == 0)
    print(f"Train positives: {n_pos}  negatives: {n_neg}  ratio: {n_neg/max(n_pos,1):.1f}:1")

    weights = [n_neg / n_pos if s['label'] == 1 else 1.0
               for s in train_dataset.samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=NUM_WORKERS,
                              pin_memory=False, persistent_workers=False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=False, persistent_workers=False)

    model     = UNet3D(in_channels=1, num_classes=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)
    scaler    = GradScaler("cuda")
    best_dice = -1.0

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        model.train()
        train_loss = 0.0
        for i, (imgs, masks, pad_mask, label) in enumerate(train_loader):
            if i == 0:
                print(f"  imgs shape:  {imgs.shape}")
                print(f"  mask sum:    {masks.sum().item():.0f}  "
                      f"(non-zero pixels in batch)")

            imgs     = imgs.to(device)
            masks    = masks.to(device)
            pad_mask = pad_mask.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                preds = model(imgs)
                loss  = bce_loss(preds, masks, pad_mask, criterion) + \
                        dice_loss(preds, masks, pad_mask)

            if torch.isnan(loss):
                print(f"  [WARNING] NaN at batch {i}, skipping")
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

        model.eval()
        val_loss = val_dice_sum = 0.0
        with torch.no_grad():
            for imgs, masks, pad_mask, label in val_loader:
                imgs     = imgs.to(device)
                masks    = masks.to(device)
                pad_mask = pad_mask.to(device)
                with autocast("cuda"):
                    preds = model(imgs)
                val_loss     += (bce_loss(preds, masks, pad_mask, criterion) +
                                 dice_loss(preds, masks, pad_mask)).item()
                val_dice_sum += dice_score(preds, masks, pad_mask).item()

        val_loss     /= len(val_loader)
        val_dice_avg  = val_dice_sum / len(val_loader)
        print(f"Val loss:  {val_loss:.4f}")
        print(f"Val Dice:  {val_dice_avg:.4f}")

        scheduler.step(val_dice_avg)

        if val_dice_avg > best_dice:
            best_dice = val_dice_avg
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Best model saved (Dice: {best_dice:.4f})")


if __name__ == "__main__":
    main()