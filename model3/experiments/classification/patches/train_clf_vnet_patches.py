"""
train_clf_patches_vnet.py

Classification training for Model 3 (VNet) on patches.
512x512 images, patches_split dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from model1.experiments.classification.patches.dataset3d_clf_patches import PatchClassificationDataset
from model3.vnet3d import MammogramVNetClassifier

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CANCER_TRAIN      = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer"
CANCER_FREE_TRAIN = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer_free"
CANCER_VAL        = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer"
CANCER_FREE_VAL   = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer_free"

EPOCHS      = 50
BATCH_SIZE  = 4
NUM_WORKERS = 8
LR          = 1e-5
FOCAL_GAMMA = 0.5
CHECKPOINT  = os.path.join(os.path.dirname(__file__), 'results', 'best_model_vnet_clf_patches.pth')

# -------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=0.5, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        return (((1 - p_t) ** self.gamma) * bce).mean()


def compute_metrics(all_logits, all_labels):
    probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    if len(np.unique(labels)) < 2:
        return {'auc': float('nan'), 'sensitivity': 0, 'specificity': 0,
                'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    auc   = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        'auc':         auc,
        'sensitivity': tp / (tp + fn + 1e-8),
        'specificity': tn / (tn + fp + 1e-8),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = PatchClassificationDataset(
        cancer_dir=CANCER_TRAIN, cancer_free_dir=CANCER_FREE_TRAIN)
    val_dataset   = PatchClassificationDataset(
        cancer_dir=CANCER_VAL,   cancer_free_dir=CANCER_FREE_VAL)

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

    model = MammogramVNetClassifier(num_views=4, embed_dim=256).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")

    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    scaler   = GradScaler("cuda")
    best_auc = -1.0

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        model.train()
        train_loss = 0.0
        for i, (views, pad_mask, labels, patients) in enumerate(train_loader):
            if i == 0:
                print(f"  views shape: {views.shape}")
                print(f"  labels:      {labels.tolist()}")

            views  = views.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(views)
                loss   = criterion(logits, labels)

            if torch.isnan(loss):
                print(f"  [WARNING] NaN at batch {i}, skipping")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            if i % 50 == 0:
                print(f"  Batch {i}/{len(train_loader)}  loss: {loss.item():.4f}  grad_norm: {total_norm:.4f}")

        train_loss /= len(train_loader)
        print(f"Train loss: {train_loss:.4f}")

        model.eval()
        val_loss, all_logits, all_labels = 0.0, [], []
        with torch.no_grad():
            for views, pad_mask, labels, patients in val_loader:
                views  = views.to(device)
                labels = labels.to(device)
                with autocast("cuda"):
                    logits = model(views)
                    loss   = criterion(logits, labels)
                val_loss   += loss.item()
                all_logits.extend(logits.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        metrics   = compute_metrics(all_logits, all_labels)

        print(f"Val loss:        {val_loss:.4f}")
        print(f"Val AUC-ROC:     {metrics['auc']:.4f}")
        print(f"Val Sensitivity: {metrics['sensitivity']:.4f}"
              f"  (TP={metrics['tp']}, FN={metrics['fn']})")
        print(f"Val Specificity: {metrics['specificity']:.4f}"
              f"  (TN={metrics['tn']}, FP={metrics['fp']})")

        scheduler.step(metrics['auc'])

        if not np.isnan(metrics['auc']) and metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Best model saved (AUC: {best_auc:.4f})")


if __name__ == "__main__":
    main()