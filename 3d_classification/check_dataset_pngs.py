import os
import numpy as np
from PIL import Image
from dataset3d_pngs import MammogramPNGDataset

CANCEROUS_PATH   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancerous"
CANCER_FREE_PATH = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\train\cancer_free"

dataset = MammogramPNGDataset(
    cancerous_dir=CANCEROUS_PATH,
    cancer_free_dir=CANCER_FREE_PATH,
)

print(f"Dataset size: {len(dataset)}")
n_cancer      = sum(1 for s in dataset.samples if s['label'] == 1)
n_cancer_free = sum(1 for s in dataset.samples if s['label'] == 0)
print(f"  Cancerous sequences:   {n_cancer}")
print(f"  Cancer-free sequences: {n_cancer_free}")
print(f"  Imbalance ratio:       {n_cancer_free / max(n_cancer, 1):.1f}:1")

# Count slices with actual tumor pixels
print("\nCounting tumor slices...")
tumor_slices  = 0
total_slices  = 0
empty_masks   = 0

for s in dataset.samples:
    if s['label'] != 1:
        continue
    for fname in s['filenames']:
        mask_path = os.path.join(s['masks_dir'], fname)
        total_slices += 1
        if os.path.exists(mask_path):
            m = np.array(Image.open(mask_path).convert('L'))
            if m.max() > 0:
                tumor_slices += 1
        else:
            empty_masks += 1

print(f"  Cancerous slices total:          {total_slices}")
print(f"  Slices with tumor pixels:        {tumor_slices}")
print(f"  Missing mask files:              {empty_masks}")

# Check first 5 samples
print("\nFirst 5 samples:")
for i in range(min(5, len(dataset))):
    imgs, masks, pad_mask, label = dataset[i]
    s = dataset.samples[i]
    print(f"\nSample {i}:")
    print(f"  files:    {s['filenames']}")
    print(f"  imgs:     {imgs.shape}  dtype: {imgs.dtype}")
    print(f"  masks:    {masks.shape}  sum: {masks.sum().item():.0f}")
    print(f"  pad_mask: {pad_mask.tolist()}")
    print(f"  label:    {int(label.item())}  ({'cancer' if label.item() == 1 else 'cancer_free'})")
    print(f"  img min/max: {imgs.min().item():.3f} / {imgs.max().item():.3f}")

print("\nAll checks passed.")