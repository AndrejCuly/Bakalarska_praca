"""
split_dataset_v2.py

Splits dataset_processed_v2 into train/val/test using the SAME patient
assignments as dataset_processed. This ensures fair comparison between
old and new preprocessing.

Reads patient IDs from dataset_processed (already split) and copies
corresponding folders from pngs_processed to dataset_processed_v2.
"""

import os
import shutil
from pathlib import Path

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
# Source: new preprocessed images (unsplit)
SOURCE_ROOT = r"C:\Users\culya\Desktop\data_bakalarka\data\pngs_processed"

# Reference: old preprocessed images (already split) — used to determine splits
REFERENCE_ROOT = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed"

# Destination: new split dataset
DEST_ROOT = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed_v2"

SPLITS  = ['train', 'val', 'test']
CLASSES = ['cancerous', 'cancer_free']

# -------------------------------------------------------------------------

def main():
    source_root    = Path(SOURCE_ROOT)
    reference_root = Path(REFERENCE_ROOT)
    dest_root      = Path(DEST_ROOT)

    total_copied  = 0
    total_missing = 0

    for split in SPLITS:
        for cls in CLASSES:
            ref_dir  = reference_root / split / cls
            dest_dir = dest_root / split / cls

            if not ref_dir.exists():
                print(f"[WARNING] Reference dir not found: {ref_dir}")
                continue

            # Get patient IDs from reference split
            patients = sorted([
                p.name for p in ref_dir.iterdir() if p.is_dir()
            ])

            print(f"\n{split}/{cls}: {len(patients)} patients")

            # Map class name to source folder name
            src_cls = 'cancerous' if cls == 'cancerous' else 'cancer_free'
            src_dir = source_root / src_cls

            copied  = 0
            missing = 0

            for patient_id in patients:
                src_patient  = src_dir / patient_id
                dest_patient = dest_dir / patient_id

                if not src_patient.exists():
                    print(f"  [MISSING] {patient_id} not found in {src_dir}")
                    missing += 1
                    continue

                if dest_patient.exists():
                    # Already copied
                    copied += 1
                    continue

                shutil.copytree(str(src_patient), str(dest_patient))
                copied += 1

            print(f"  Copied: {copied}  Missing: {missing}")
            total_copied  += copied
            total_missing += missing

    print(f"\nDone. Total copied: {total_copied}, missing: {total_missing}")
    print(f"Output: {dest_root}")


if __name__ == "__main__":
    main()