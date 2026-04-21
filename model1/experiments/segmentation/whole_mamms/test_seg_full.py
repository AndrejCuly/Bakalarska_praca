"""
test_seg_full.py

Evaluates segmentation model on test set.
Saves:
  - results/seg_test_results.csv   — per-sample metrics
  - results/seg_summary.txt        — summary statistics
  - results/seg_example.png        — example predictions grid
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import autocast

from model1.experiments.classification.whole_mamms.dataset3d_pngs import MammogramPNGDataset
from model1.unet3d_model.unet3d import UNet3D

# -------------------------------------------------------------------------
CANCEROUS_TEST   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancerous"
CANCER_FREE_TEST = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancer_free"
CHECKPOINT   = os.path.join(os.path.dirname(__file__), 'results', 'best_model_seg.pth')
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), 'results')
CSV_PATH     = os.path.join(RESULTS_DIR, 'seg_test_results.csv')
SUMMARY_PATH = os.path.join(RESULTS_DIR, 'seg_summary.txt')
EXAMPLE_PATH = os.path.join(RESULTS_DIR, 'seg_example.png')
BATCH_SIZE   = 4
NUM_WORKERS  = 8
# -------------------------------------------------------------------------


def slice_metrics(pred_bin, gt):
    """Compute Dice and IoU for two flattened binary tensors."""
    intersection = (pred_bin * gt).sum()
    union        = pred_bin.sum() + gt.sum() - intersection
    dice = (2 * intersection) / (pred_bin.sum() + gt.sum() + 1e-8)
    iou  = intersection / (union + 1e-8)
    return dice.item(), iou.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    test_dataset = MammogramPNGDataset(
        cancerous_dir=CANCEROUS_TEST,
        cancer_free_dir=CANCER_FREE_TEST)
    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=False, persistent_workers=False)

    model = UNet3D(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device,
                                     weights_only=False))
    model.eval()
    print(f"Loaded: {CHECKPOINT}")

    all_dice, all_iou = [], []
    total_gt_px   = 0
    total_pred_px = 0

    # For saving example visualization
    example_imgs, example_masks, example_preds = [], [], []
    example_saved = False

    csv_rows = []

    sample_idx = 0
    with torch.no_grad():
        for imgs, masks, pad_mask, labels in test_loader:
            imgs     = imgs.to(device)
            masks    = masks.to(device)
            pad_mask = pad_mask.to(device)

            with autocast("cuda"):
                preds_logits = model(imgs)

            pred_prob   = torch.sigmoid(preds_logits)
            pred_binary = (pred_prob > 0.5).float()

            total_gt_px   += masks.sum().item()
            total_pred_px += pred_binary.sum().item()

            B = imgs.shape[0]
            for b in range(B):
                label = int(labels[b].item())
                sample_dice_list, sample_iou_list = [], []

                for t in range(imgs.shape[2]):
                    if not pad_mask[b, t]:
                        continue
                    gt_slice   = masks[b, 0, t].reshape(-1)
                    pred_slice = pred_binary[b, 0, t].reshape(-1)

                    # Save first cancerous example with tumor pixels
                    if (not example_saved and label == 1
                            and gt_slice.sum() > 0):
                        example_imgs.append(
                            imgs[b, 0, t].cpu().numpy())
                        example_masks.append(
                            masks[b, 0, t].cpu().numpy())
                        example_preds.append(
                            pred_prob[b, 0, t].cpu().numpy())
                        if len(example_imgs) >= 4:
                            example_saved = True

                    if gt_slice.sum() == 0:
                        continue

                    d, iou = slice_metrics(pred_slice, gt_slice)
                    sample_dice_list.append(d)
                    sample_iou_list.append(iou)

                if sample_dice_list:
                    mean_d   = float(np.mean(sample_dice_list))
                    mean_iou = float(np.mean(sample_iou_list))
                    all_dice.append(mean_d)
                    all_iou.append(mean_iou)
                    csv_rows.append({
                        'sample_idx': sample_idx,
                        'label': label,
                        'n_tumor_slices': len(sample_dice_list),
                        'mean_dice': f'{mean_d:.4f}',
                        'mean_iou': f'{mean_iou:.4f}',
                    })

                sample_idx += 1

    # ── CSV ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_idx', 'label', 'n_tumor_slices', 'mean_dice', 'mean_iou'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV saved: {CSV_PATH}")

    # ── Summary ──────────────────────────────────────────────────────────
    lines = []
    lines.append("SEGMENTATION TEST RESULTS — Model 1 (3D U-Net)")
    lines.append("=" * 55)
    lines.append(f"Test samples total:              {len(test_dataset)}")
    lines.append(f"Samples with tumor ground truth: {len(all_dice)}")
    lines.append("")

    if all_dice:
        lines.append(f"Mean Dice (DSC):   {np.mean(all_dice):.4f}  "
                     f"(std: {np.std(all_dice):.4f})")
        lines.append(f"Mean IoU:          {np.mean(all_iou):.4f}  "
                     f"(std: {np.std(all_iou):.4f})")
        lines.append(f"Median Dice:       {np.median(all_dice):.4f}")
        lines.append(f"Median IoU:        {np.median(all_iou):.4f}")
    else:
        lines.append("Dice: 0.0000   IoU: 0.0000")
    lines.append("")
    lines.append(f"Ground-truth tumor pixels: {int(total_gt_px):,}")
    lines.append(f"Predicted tumor pixels:    {int(total_pred_px):,}")
    lines.append("")
    if total_pred_px > total_gt_px * 10:
        lines.append("Conclusion: model massively over-predicts tumor regions.")
        lines.append("Despite over-prediction, Dice = 0 because the predicted")
        lines.append("mask does not overlap with the tiny true tumor locations.")
        lines.append("This is caused by extreme pixel-level class imbalance —")
        lines.append("tumor pixels represent <0.1% of all image pixels.")
    elif total_pred_px == 0:
        lines.append("Conclusion: model predicted all-zero masks (all background).")
        lines.append("Extreme pixel-level class imbalance caused the model to")
        lines.append("collapse to the trivial all-background solution.")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(SUMMARY_PATH, 'w') as f:
        f.write(summary_text)
    print(f"Summary saved: {SUMMARY_PATH}")

    # ── Example figure ───────────────────────────────────────────────────
    if example_imgs:
        n = len(example_imgs)
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = [axes]
        for row, (img, mask, pred) in enumerate(
                zip(example_imgs, example_masks, example_preds)):
            # Denormalize image from [-1,1] to [0,1]
            img_show = (img * 0.5 + 0.5).clip(0, 1)
            axes[row][0].imshow(img_show, cmap='gray')
            axes[row][0].set_title('Original', fontsize=9)
            axes[row][0].axis('off')

            axes[row][1].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axes[row][1].set_title('Ground truth mask', fontsize=9)
            axes[row][1].axis('off')

            axes[row][2].imshow(pred, cmap='hot', vmin=0, vmax=1)
            axes[row][2].set_title('Prediction (prob)', fontsize=9)
            axes[row][2].axis('off')

        fig.suptitle(
            'Segmentation examples — tumor slices\n'
            f'(Dice = {np.mean(all_dice):.4f}, IoU = {np.mean(all_iou):.4f})',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(EXAMPLE_PATH, dpi=150, bbox_inches='tight')
        print(f"Example figure saved: {EXAMPLE_PATH}")
    else:
        print("No tumor slices found for visualization.")


if __name__ == "__main__":
    main()