"""
cache_dataset.py

Pre-caches all PNG images in the dataset as .pt tensor files.
Run this ONCE before training. After caching, training loads
torch.load() instead of PIL decode + resize + normalize,
which is 3-4x faster per batch.

What it does:
- Reads every .png file in dataset_processed/
- Applies the same transform as the dataset class (resize to 256x256,
  normalize to [-1, 1])
- Saves the result as a .pt file next to the original .png

Example:
    00075_20990909_L_CC_1.png  →  00075_20990909_L_CC_1.pt

The dataset class will automatically use .pt files if they exist.
"""

import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import time

# -------------------------------------------------------------------------
# Config — set this to your dataset_processed root folder
# -------------------------------------------------------------------------
DATASET_ROOT = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed"
IMG_SIZE     = 256

# -------------------------------------------------------------------------

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE),
                      interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def cache_directory(root):
    """Walk directory tree and cache all .png files as .pt tensors."""
    png_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.png'):
                png_files.append(os.path.join(dirpath, fname))

    total   = len(png_files)
    cached  = 0
    skipped = 0
    errors  = 0

    print(f"Found {total:,} PNG files to cache")
    start = time.time()

    for i, png_path in enumerate(png_files):
        pt_path = png_path.replace('.png', '.pt')

        # Skip if already cached
        if os.path.exists(pt_path):
            skipped += 1
            continue

        try:
            img = Image.open(png_path).convert('L')
            tensor = img_transform(img)   # [1, 256, 256]
            torch.save(tensor, pt_path)
            cached += 1
        except Exception as e:
            print(f"  ERROR: {png_path} — {e}")
            errors += 1

        # Progress every 1000 files
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate    = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            print(f"  {i+1:,}/{total:,} — "
                  f"{rate:.0f} files/sec — "
                  f"~{remaining/60:.1f} min remaining")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"  Cached:  {cached:,}")
    print(f"  Skipped: {skipped:,} (already existed)")
    print(f"  Errors:  {errors:,}")


if __name__ == "__main__":
    print(f"Caching dataset at: {DATASET_ROOT}")
    print("This will take 15-25 minutes. Run once, then training is 3-4x faster.\n")
    cache_directory(DATASET_ROOT)
    print("\nCaching complete. You can now run training.")