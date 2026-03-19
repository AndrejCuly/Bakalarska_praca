"""
train_clf.py

Training script for per-patient binary classification using the
shared-weight 3D CNN encoder (MammogramClassifier).

Key differences from segmentation training:
- One label per patient (cancerous / cancer_free)
- Focal Loss instead of BCE+Dice — better for 29:1 imbalance
- Metrics: AUC-ROC, accuracy, sensitivity, specificity
- No pad_mask needed in loss (classification is patient-level)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np

from experiments.classification.whole_mamms.dataset3d_clf import MammogramClassificationDataset
from experiments.classification.classifier3d import MammogramClassifier

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
LR          = 1e-4
CHECKPOINT  = "best_model_clf.pth"

# Focal loss gamma — higher = more focus on hard examples
FOCAL_GAMMA = 2.0

# -------------------------------------------------------------------------
# Focal Loss
# -------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.
    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Better than BCE for heavily imbalanced datasets because it
    down-weights easy negatives and focuses training on hard examples.
    """

    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        prob     = torch.sigmoid(logits)
        p_t      = prob * targets + (1 - prob) * (1 - targets)
        focal_w  = (1 - p_t) ** self.gamma
        loss     = focal_w * bce
        return loss.mean()


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def compute_metrics(all_logits, all_labels):
    """
    Compute AUC-ROC, accuracy, sensitivity, specificity from collected
    logits and labels across the validation set.
    """
    probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)

    # AUC-ROC — needs both classes present
    if len(np.unique(labels)) < 2:
        auc = float('nan')
    else:
        auc = roc_auc_score(labels, probs)

    # Hard predictions at 0.5 threshold
    preds = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel() \
        if len(np.unique(labels)) > 1 else (0, 0, 0, 0)

    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)   # recall for positive class
    specificity = tn / (tn + fp + 1e-8)

    return {
        'auc':         auc,
        'accuracy':    accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': int(tp), 'tn': int(tn),
        'fp': int(fp), 'fn': int(fn),
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- Datasets ---
    train_dataset = MammogramClassificationDataset(
        cancerous_dir=CANCEROUS_TRAIN,
        cancer_free_dir=CANCER_FREE_TRAIN,
    )
    val_dataset = MammogramClassificationDataset(
        cancerous_dir=CANCEROUS_VAL,
        cancer_free_dir=CANCER_FREE_VAL,
    )

    print(f"Train patients: {len(train_dataset)}")
    print(f"Val patients:   {len(val_dataset)}")

    n_pos = sum(1 for s in train_dataset.samples if s['label'] == 1)
    n_neg = sum(1 for s in train_dataset.samples if s['label'] == 0)
    print(f"Train positives: {n_pos}  negatives: {n_neg}  ratio: {n_neg/max(n_pos,1):.1f}:1")

    # --- Imbalance: WeightedRandomSampler ---
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
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Batches per epoch: {len(train_loader)}")

    # --- Model ---
    model = MammogramClassifier(num_views=4, embed_dim=256).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Focal loss with pos_weight for extra imbalance handling
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion  = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    scaler = GradScaler("cuda")

    best_auc = -1.0

    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

        # ---- Training ----
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
                print(f"  [WARNING] NaN loss at batch {i}, skipping")
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if i % 50 == 0:
                print(f"  Batch {i}/{len(train_loader)}  loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        print(f"Train loss: {train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss    = 0.0
        all_logits  = []
        all_labels  = []

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
        print(f"Val Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Val Sensitivity: {metrics['sensitivity']:.4f}  (TP={metrics['tp']}, FN={metrics['fn']})")
        print(f"Val Specificity: {metrics['specificity']:.4f}  (TN={metrics['tn']}, FP={metrics['fp']})")

        scheduler.step(metrics['auc'])

        if not np.isnan(metrics['auc']) and metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"  ✓ Best model saved (AUC: {best_auc:.4f})")


if __name__ == "__main__":
    main()