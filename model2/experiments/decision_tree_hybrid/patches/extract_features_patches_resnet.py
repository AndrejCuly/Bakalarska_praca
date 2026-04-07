"""
extract_features_patches_resnet.py

Extracts 256-dimensional feature vectors from the trained Model 2
ResNet-10 patches encoder for all patch samples in train, val, and test sets.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast

from model1.experiments.classification.patches.dataset3d_clf_patches import PatchClassificationDataset
from model2.resnet3d import MammogramResNetClassifier

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
CANCER_TRAIN      = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer"
CANCER_FREE_TRAIN = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer_free"
CANCER_VAL        = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer"
CANCER_FREE_VAL   = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer_free"
CANCER_TEST       = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\test\cancer"
CANCER_FREE_TEST  = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\test\cancer_free"

CHECKPOINT  = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model2\experiments\classification\patches\results\best_model_resnet_clf_patches.pth"
BATCH_SIZE  = 4
NUM_WORKERS = 8
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'features_patches')

# -------------------------------------------------------------------------

class EncoderOnly(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.encoder   = full_model.encoder
        self.num_views = full_model.num_views

    def forward(self, views):
        B, V, C, T, H, W = views.shape
        x = views.view(B * V, C, T, H, W)
        x = self.encoder(x)
        x = x.view(B, V, -1)
        x = x.mean(dim=1)
        return x


def extract_split(encoder, cancer_dir, cancer_free_dir,
                  split_name, device, output_dir):
    dataset = PatchClassificationDataset(
        cancer_dir=cancer_dir,
        cancer_free_dir=cancer_free_dir,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=False,
    )

    n_pos = sum(1 for s in dataset.samples if s['label'] == 1)
    n_neg = sum(1 for s in dataset.samples if s['label'] == 0)
    print(f"\n{split_name}: {len(dataset)} samples ({n_pos} tumor, {n_neg} normal)")

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
            # Filter NaN features
            valid = ~torch.isnan(features).any(dim=1).cpu()
            all_features.append(features[valid.to(features.device)].cpu().numpy())
            all_labels.extend(labels[valid].numpy().tolist())
            all_patients.extend([p for p, v in zip(patients, valid.tolist()) if v])

    features_arr    = np.concatenate(all_features, axis=0).astype(np.float32)
    labels_arr      = np.array(all_labels, dtype=np.int32)
    patients_arr    = np.array(all_patients)
    patch_types_arr = np.array(all_patch_types[:len(labels_arr)])

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

    full_model = MammogramResNetClassifier(num_views=4, embed_dim=256)
    full_model.load_state_dict(torch.load(CHECKPOINT, map_location=device,
                                          weights_only=False))
    full_model.to(device)
    print(f"Loaded weights from {CHECKPOINT}")

    encoder = EncoderOnly(full_model).to(device)

    extract_split(encoder, CANCER_TRAIN, CANCER_FREE_TRAIN, "train", device, OUTPUT_DIR)
    extract_split(encoder, CANCER_VAL,   CANCER_FREE_VAL,   "val",   device, OUTPUT_DIR)
    extract_split(encoder, CANCER_TEST,  CANCER_FREE_TEST,  "test",  device, OUTPUT_DIR)

    print("\nFeature extraction complete.")
    print(f"Files saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()