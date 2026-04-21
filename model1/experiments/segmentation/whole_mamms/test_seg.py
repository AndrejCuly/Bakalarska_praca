"""
test_seg.py

Evaluates the trained segmentation model on the test set.
Reports Dice coefficient and IoU (Jaccard index) — both expected
to be near 0 due to extreme pixel-level class imbalance.

Place in: model1/experiments/segmentation/whole_mamms/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast

from model1.experiments.classification.whole_mamms.dataset3d_pngs import MammogramPNGDataset
from model1.unet3d_model.unet3d import UNet3D

# -------------------------------------------------------------------------
CANCEROUS_TEST   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancerous"
CANCER_FREE_TEST = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancer_free"
CHECKPOINT  = os.path.join(os.path.dirname(__file__), 'results', 'best_model_seg.pth')
BATCH_SIZE  = 4
NUM_WORKERS = 8
# -------------------------------------------------------------------------


def compute_metrics(pred_logits, target, pad_mask):
    """
    Compute per-sample Dice and IoU over non-padded tumor slices.
    Returns lists of (dice, iou) for each sample that has ground-truth tumor pixels.
    """
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    B, C, T, H, W = pred.shape

    dice_scores = []
    iou_scores  = []

    for b in range(B):
        sample_dice = []
        sample_iou  = []
        for t in range(T):
            if not pad_mask[b, t]:
                continue
            p = pred[b, 0, t].reshape(-1)
            g = target[b, 0, t].reshape(-1)
            if g.sum() == 0:
                continue  # skip slices with no tumor in ground truth
            intersection = (p * g).sum()
            union        = p.sum() + g.sum() - intersection
            dice = (2 * intersection) / (p.sum() + g.sum() + 1e-8)
            iou  = intersection / (union + 1e-8)
            sample_dice.append(dice.item())
            sample_iou.append(iou.item())

        if sample_dice:
            dice_scores.append(np.mean(sample_dice))
            iou_scores.append(np.mean(sample_iou))

    return dice_scores, iou_scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_dataset = MammogramPNGDataset(
        cancerous_dir=CANCEROUS_TEST,
        cancer_free_dir=CANCER_FREE_TEST)
    print(f"Test samples: {len(test_dataset)}")
    n_pos = sum(1 for s in test_dataset.samples if s['label'] == 1)
    n_neg = sum(1 for s in test_dataset.samples if s['label'] == 0)
    print(f"  Cancerous sequences:   {n_pos}")
    print(f"  Cancer-free sequences: {n_neg}")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=False, persistent_workers=False)

    model = UNet3D(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device,
                                     weights_only=False))
    model.eval()
    print(f"Loaded weights from {CHECKPOINT}")

    all_dice, all_iou = [], []
    total_pred_positive  = 0
    total_true_positive_pixels = 0
    total_pred_positive_pixels = 0

    with torch.no_grad():
        for imgs, masks, pad_mask, label in test_loader:
            imgs     = imgs.to(device)
            masks    = masks.to(device)
            pad_mask = pad_mask.to(device)

            with autocast("cuda"):
                preds = model(imgs)

            pred_binary = (torch.sigmoid(preds) > 0.5).float()
            total_pred_positive_pixels  += pred_binary.sum().item()
            total_true_positive_pixels  += masks.sum().item()

            dice_batch, iou_batch = compute_metrics(preds, masks, pad_mask)
            all_dice.extend(dice_batch)
            all_iou.extend(iou_batch)

    print("\n" + "="*55)
    print("TEST SET RESULTS — Model 1 Segmentation (Whole Mammograms)")
    print("="*55)

    if all_dice:
        print(f"Samples with tumor ground truth: {len(all_dice)}")
        print(f"Mean Dice (DSC):  {np.mean(all_dice):.4f}  "
              f"(std: {np.std(all_dice):.4f})")
        print(f"Mean IoU:         {np.mean(all_iou):.4f}  "
              f"(std: {np.std(all_iou):.4f})")
        print(f"Median Dice:      {np.median(all_dice):.4f}")
        print(f"Median IoU:       {np.median(all_iou):.4f}")
    else:
        print("No tumor slices found in test set — "
              "model predicted all zeros (Dice = 0.0000, IoU = 0.0000)")

    print(f"\nTotal ground-truth tumor pixels: {int(total_true_positive_pixels)}")
    print(f"Total predicted tumor pixels:    {int(total_pred_positive_pixels)}")

    if total_pred_positive_pixels == 0:
        print("\nConclusion: model predicted all-zero masks on entire test set.")
        print("This is consistent with extreme pixel-level class imbalance —")
        print("tumor pixels represent a tiny fraction of all pixels, so the")
        print("model learns to predict background for all inputs.")


if __name__ == "__main__":
    main()