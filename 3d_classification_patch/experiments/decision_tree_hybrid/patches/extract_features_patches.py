"""
extract_features_patches.py

Extracts 256-dimensional feature vectors from the trained patches
encoder (best_model_clf_patches.pth) for all patch samples in
train, val, and test sets.

Output: three .npz files saved to features_patches/ folder:
    - features: [N, 256] float32 array
    - labels:   [N] int array (1=tumor patch, 0=normal patch)
    - patients: [N] string array
    - patch_types: [N] string array (tumor/normal_0/plain)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast

from experiments.classification.patches.dataset3d_clf_patches import PatchClassificationDataset
from experiments.classification.classifier3d import MammogramClassifier

# -------------------------------------------------------------------------
# Config — update paths to match your system
# -------------------------------------------------------------------------
CANCER_TRAIN      = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer"
CANCER_FREE_TRAIN = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer_free"
CANCER_VAL        = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer"
CANCER_FREE_VAL   = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer_free"
CANCER_TEST       = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\test\cancer"
CANCER_FREE_TEST  = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\test\cancer_free"

CHECKPOINT = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\experiments\classification\patches\results\best_model_clf_patches.pth"
BATCH_SIZE   = 8
NUM_WORKERS  = 4
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'features_patches')

# -------------------------------------------------------------------------


class EncoderOnly(torch.nn.Module):
    """
    Extracts the 256-dim patient embedding from MammogramClassifier,
    before the final classification layer.
    """
    def __init__(self, full_model):
        super().__init__()
        self.encoder   = full_model.encoder
        self.num_views = full_model.num_views

    def forward(self, views):
        B, V, C, T, H, W = views.shape
        x = views.view(B * V, C, T, H, W)
        x = self.encoder(x)       # [B*V, embed_dim]
        x = x.view(B, V, -1)      # [B, V, embed_dim]
        x = x.mean(dim=1)         # [B, embed_dim]
        return x


def extract_split(encoder, cancer_dir, cancer_free_dir,
                  split_name, device, output_dir):
    dataset = PatchClassificationDataset(
        cancer_dir=cancer_dir,
        cancer_free_dir=cancer_free_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
    )

    n_pos = sum(1 for s in dataset.samples if s['label'] == 1)
    n_neg = sum(1 for s in dataset.samples if s['label'] == 0)
    print(f"\n{split_name}: {len(dataset)} samples "
          f"({n_pos} tumor, {n_neg} normal)")

    all_features    = []
    all_labels      = []
    all_patients    = []
    all_patch_types = [s['patch_type'] for s in dataset.samples]

    encoder.eval()
    with torch.no_grad():
        for views, pad_mask, labels, patients in loader:
            views = views.to(device)
            with autocast("cuda"):
                features = encoder(views)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_patients.extend(patients)

    features_arr    = np.concatenate(all_features, axis=0).astype(np.float32)
    labels_arr      = np.array(all_labels, dtype=np.int32)
    patients_arr    = np.array(all_patients)
    patch_types_arr = np.array(all_patch_types)

    print(f"  Feature matrix: {features_arr.shape}")
    print(f"  Positive (tumor) samples: {labels_arr.sum()}")

    out_path = os.path.join(output_dir, f"{split_name}_features.npz")
    np.savez(out_path,
             features=features_arr,
             labels=labels_arr,
             patients=patients_arr,
             patch_types=patch_types_arr)
    print(f"  Saved to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_model = MammogramClassifier(num_views=4, embed_dim=256)
    full_model.load_state_dict(
        torch.load(CHECKPOINT, map_location=device, weights_only=True)
    )
    full_model.to(device)
    print(f"Loaded weights from {CHECKPOINT}")

    encoder = EncoderOnly(full_model).to(device)

    extract_split(encoder, CANCER_TRAIN, CANCER_FREE_TRAIN,
                  "train", device, OUTPUT_DIR)
    extract_split(encoder, CANCER_VAL, CANCER_FREE_VAL,
                  "val", device, OUTPUT_DIR)
    extract_split(encoder, CANCER_TEST, CANCER_FREE_TEST,
                  "test", device, OUTPUT_DIR)

    print("\nFeature extraction complete.")
    print(f"Files saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()