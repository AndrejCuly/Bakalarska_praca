"""
test_clf_vnet.py

Evaluates best_model_vnet_clf.pth on held-out test set.
Model 3 (VNet) — whole mammograms, 512x512.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import numpy as np
import csv
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model1.experiments.classification.whole_mamms.dataset3d_clf import MammogramClassificationDataset
from model3.vnet3d import MammogramVNetClassifier

# -------------------------------------------------------------------------
CANCEROUS_TEST   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancerous"
CANCER_FREE_TEST = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancer_free"
CHECKPOINT  = os.path.join(os.path.dirname(__file__), 'results', 'best_model_vnet_clf.pth')
BATCH_SIZE  = 4
NUM_WORKERS = 8
RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'results', 'test_results.csv')
ROC_PLOT    = os.path.join(os.path.dirname(__file__), 'results', 'roc_curve.png')
# -------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    test_dataset = MammogramClassificationDataset(
        cancerous_dir=CANCEROUS_TEST, cancer_free_dir=CANCER_FREE_TEST)
    print(f"Test patients: {len(test_dataset)}")
    n_pos = sum(1 for s in test_dataset.samples if s['label'] == 1)
    n_neg = sum(1 for s in test_dataset.samples if s['label'] == 0)
    print(f"  Cancerous:   {n_pos}")
    print(f"  Cancer-free: {n_neg}")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=False, persistent_workers=False)

    model = MammogramVNetClassifier(num_views=4, embed_dim=256).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device,
                                     weights_only=False))
    model.eval()
    print(f"Loaded weights from {CHECKPOINT}")

    all_logits, all_labels, all_patients = [], [], []
    with torch.no_grad():
        for views, pad_mask, labels, patients in test_loader:
            views = views.to(device)
            with autocast("cuda"):
                logits = model(views)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_patients.extend(patients)

    probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv         = tp / (tp + fp + 1e-8)
    npv         = tn / (tn + fn + 1e-8)

    print("\n" + "="*50)
    print("TEST SET RESULTS — Model 3 VNet (Whole Mammograms)")
    print("="*50)
    print(f"AUC-ROC:      {auc:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}  (TP={int(tp)}, FN={int(fn)})")
    print(f"Specificity:  {specificity:.4f}  (TN={int(tn)}, FP={int(fp)})")
    print(f"PPV:          {ppv:.4f}")
    print(f"NPV:          {npv:.4f}")
    print(f"\nConfusion matrix:")
    print(f"              Predicted 0   Predicted 1")
    print(f"  Actual 0       {int(tn):5d}         {int(fp):5d}")
    print(f"  Actual 1       {int(fn):5d}         {int(tp):5d}")

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='steelblue', lw=2,
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve — Model 3 VNet (Whole Mammograms)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(ROC_PLOT), exist_ok=True)
    plt.savefig(ROC_PLOT, dpi=150)
    print(f"\nROC curve saved to {ROC_PLOT}")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['patient_id', 'true_label', 'predicted_prob', 'predicted_label'])
        for patient, label, prob, pred in zip(all_patients, all_labels, probs, preds):
            writer.writerow([patient, int(label), f'{prob:.4f}', int(pred)])
    print(f"Per-patient results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()