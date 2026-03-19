"""
extract_features.py

Extracts 256-dimensional feature vectors from the trained whole-mammogram
encoder (best_model_clf.pth) for all patients in train, val, and test sets.

The encoder is used as a frozen feature extractor — we take the output of
the mean-pooled view embeddings BEFORE the final classification layer.

Output: three .npz files (one per split) each containing:
    - features: [N, 256] float32 array
    - labels:   [N] int array (1=cancerous, 0=cancer_free)
    - patients: [N] string array (patient IDs)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast

from experiments.classification.whole_mamms.dataset3d_clf import MammogramClassificationDataset
from experiments.classification.classifier3d import MammogramClassifier

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CANCEROUS_TRAIN   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancerous"
CANCER_FREE_TRAIN = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancer_free"
CANCEROUS_VAL     = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val\cancerous"
CANCER_FREE_VAL   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val\cancer_free"
CANCEROUS_TEST    = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancerous"
CANCER_FREE_TEST  = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancer_free"

CHECKPOINT   = r"experiments\classification\whole_mamms\results\best_model_clf.pth"
BATCH_SIZE   = 8
NUM_WORKERS  = 12
OUTPUT_DIR   = "features"

# -------------------------------------------------------------------------


class EncoderOnly(torch.nn.Module):
    """
    Wraps MammogramClassifier to return the 256-dim embedding
    instead of the final logit. This is the feature vector we
    feed to the classical ML classifiers.
    """
    def __init__(self, full_model):
        super().__init__()
        self.encoder    = full_model.encoder
        self.num_views  = full_model.num_views

    def forward(self, views):
        B, V, C, T, H, W = views.shape
        x = views.view(B * V, C, T, H, W)
        x = self.encoder(x)           # [B*V, embed_dim]
        x = x.view(B, V, -1)          # [B, V, embed_dim]
        x = x.mean(dim=1)             # [B, embed_dim]  ← feature vector
        return x


def extract_split(encoder, cancerous_dir, cancer_free_dir, split_name, device, output_dir):
    dataset = MammogramClassificationDataset(
        cancerous_dir=cancerous_dir,
        cancer_free_dir=cancer_free_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    n_pos = sum(1 for s in dataset.samples if s['label'] == 1)
    n_neg = sum(1 for s in dataset.samples if s['label'] == 0)
    print(f"\n{split_name}: {len(dataset)} patients "
          f"({n_pos} cancerous, {n_neg} cancer-free)")

    all_features = []
    all_labels   = []
    all_patients = []

    encoder.eval()
    with torch.no_grad():
        for views, pad_mask, labels, patients in loader:
            views = views.to(device)
            with autocast("cuda"):
                features = encoder(views)   # [B, 256]
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_patients.extend(patients)

    features_arr = np.concatenate(all_features, axis=0).astype(np.float32)
    labels_arr   = np.array(all_labels, dtype=np.int32)
    patients_arr = np.array(all_patients)

    print(f"  Feature matrix: {features_arr.shape}")
    print(f"  Positive samples: {labels_arr.sum()}")

    out_path = os.path.join(output_dir, f"{split_name}_features.npz")
    np.savez(out_path,
             features=features_arr,
             labels=labels_arr,
             patients=patients_arr)
    print(f"  Saved to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load full model
    full_model = MammogramClassifier(num_views=4, embed_dim=256)
    full_model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    full_model.to(device)
    print(f"Loaded weights from {CHECKPOINT}")

    # Wrap to extract features only
    encoder = EncoderOnly(full_model).to(device)

    # Extract for all three splits
    extract_split(encoder, CANCEROUS_TRAIN, CANCER_FREE_TRAIN,
                  "train", device, OUTPUT_DIR)
    extract_split(encoder, CANCEROUS_VAL,   CANCER_FREE_VAL,
                  "val",   device, OUTPUT_DIR)
    extract_split(encoder, CANCEROUS_TEST,  CANCER_FREE_TEST,
                  "test",  device, OUTPUT_DIR)

    print("\nFeature extraction complete.")
    print(f"Files saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()