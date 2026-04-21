"""
extract_features_resnet50.py

Extracts 256-dim feature vectors from the frozen ResNet50 3D encoder
for whole mammogram classification (Model 4).

Run this BEFORE train_hybrid_resnet50.py.
Saves features to model4/experiments/decision_tree_hybrid/whole_mamms/features/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import torch
import numpy as np
from torch.utils.data import DataLoader

from model1.experiments.classification.whole_mamms.dataset3d_clf import MammogramClassificationDataset
from model4.resnet50_3d import MammogramResNet50Classifier

# -------------------------------------------------------------------------
CANCEROUS_TRAIN      = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancerous"
CANCER_FREE_TRAIN    = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancer_free"
CANCEROUS_VAL        = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val\cancerous"
CANCER_FREE_VAL      = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\val\cancer_free"
CANCEROUS_TEST       = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancerous"
CANCER_FREE_TEST     = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancer_free"

CHECKPOINT  = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model4\experiments\classification\whole_mamms\results\best_model_resnet50_clf.pth"
FEATURES_DIR = os.path.join(os.path.dirname(__file__), 'features')
# -------------------------------------------------------------------------


class ResNet50Encoder(torch.nn.Module):
    """Wraps full model, returns 256-dim features before the classifier head."""
    def __init__(self, full_model):
        super().__init__()
        self.encoder = full_model.encoder  # already includes GAP + proj internally

    def forward(self, x):
        # x: [B, V, 1, T, H, W] — dataset includes channel dim
        B, V, C, T, H, W = x.shape
        feats = []
        for v in range(V):
            view = x[:, v, :, :, :, :]  # [B, 1, T, H, W]
            f = self.encoder(view)  # [B, embed_dim]
            feats.append(f)
        return torch.stack(feats, dim=1).mean(dim=1)  # [B, embed_dim]


def extract(loader, model, device, split_name):
    model.eval()
    all_features, all_labels, all_patients = [], [], []
    with torch.no_grad():
        for views, pad_mask, labels, patients in loader:
            views = views.to(device)
            feats = model(views)
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_patients.extend(patients)
    features = np.vstack(all_features)
    labels   = np.array(all_labels)
    print(f"  {split_name}: {features.shape[0]} patients, {features.shape[1]} dims")
    return features, labels, all_patients


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(FEATURES_DIR, exist_ok=True)

    full_model = MammogramResNet50Classifier(num_views=4, embed_dim=256)
    full_model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=False))
    full_model.to(device)
    print(f"Loaded checkpoint: {CHECKPOINT}")

    encoder = ResNet50Encoder(full_model).to(device)

    splits = [
        ("train", CANCEROUS_TRAIN, CANCER_FREE_TRAIN),
        ("val",   CANCEROUS_VAL,   CANCER_FREE_VAL),
        ("test",  CANCEROUS_TEST,  CANCER_FREE_TEST),
    ]

    for split_name, canc_dir, free_dir in splits:
        dataset = MammogramClassificationDataset(
            cancerous_dir=canc_dir, cancer_free_dir=free_dir)
        loader = DataLoader(dataset, batch_size=4, shuffle=False,
                            num_workers=4, pin_memory=False)
        feats, labels, patients = extract(loader, encoder, device, split_name)
        np.save(os.path.join(FEATURES_DIR, f"{split_name}_features.npy"), feats)
        np.save(os.path.join(FEATURES_DIR, f"{split_name}_labels.npy"),   labels)
        np.save(os.path.join(FEATURES_DIR, f"{split_name}_patients.npy"), np.array(patients))

    print(f"\nFeatures saved to: {FEATURES_DIR}")


if __name__ == "__main__":
    main()