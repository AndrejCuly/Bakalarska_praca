"""
extract_features_vnet_patches.py

Extracts 256-dim feature vectors from the frozen VNet encoder
for patches classification (Model 3).

Run this BEFORE train_hybrid_vnet_patches.py.
Saves features to: decision_tree_hybrid/patches/features_patches/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast

from model1.experiments.classification.patches.dataset3d_clf_patches import PatchClassificationDataset
from model3.vnet3d import MammogramVNetClassifier

# -------------------------------------------------------------------------
CANCER = {
    'train': r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer",
    'val':   r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer",
    'test':  r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\test\cancer",
}
CANCER_FREE = {
    'train': r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\train\cancer_free",
    'val':   r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\val\cancer_free",
    'test':  r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\test\cancer_free",
}
CHECKPOINT   = os.path.join(os.path.dirname(__file__), '..', '..', 'classification',
                             'patches', 'results', 'best_model_vnet_clf_patches.pth')
FEATURES_DIR = os.path.join(os.path.dirname(__file__), 'features_patches')
BATCH_SIZE   = 4
NUM_WORKERS  = 8
# -------------------------------------------------------------------------

class VNetFeatureExtractor(torch.nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.encoder = full_model.encoder

    def forward(self, views):
        feats = []
        for v in range(views.shape[1]):
            x = views[:, v, :, :, :, :]
            feats.append(self.encoder(x))
        return torch.stack(feats, dim=1).mean(dim=1)


def extract_split(extractor, split, device):
    dataset = PatchClassificationDataset(
        cancer_dir=CANCER[split],
        cancer_free_dir=CANCER_FREE[split])
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)

    all_features, all_labels, all_patients = [], [], []
    extractor.eval()
    with torch.no_grad():
        for views, pad_mask, labels, patients in loader:
            views = views.to(device)
            with autocast("cuda"):
                feats = extractor(views)
            all_features.append(feats.cpu().float().numpy())
            all_labels.extend(labels.numpy())
            all_patients.extend(patients)

    features = np.concatenate(all_features, axis=0)
    labels   = np.array(all_labels)
    patients = np.array(all_patients)
    print(f"  {split}: {features.shape}  positives: {labels.sum()}")
    return features, labels, patients


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MammogramVNetClassifier(num_views=4, embed_dim=256)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=False))
    model = model.to(device)

    extractor = VNetFeatureExtractor(model)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    for split in ['train', 'val', 'test']:
        print(f"\nExtracting {split}...")
        features, labels, patients = extract_split(extractor, split, device)
        out_path = os.path.join(FEATURES_DIR, f"{split}_features.npz")
        np.savez(out_path, features=features, labels=labels, patients=patients)
        print(f"  Saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()