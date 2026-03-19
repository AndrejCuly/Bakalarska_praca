"""
fix_masks.py

Regenerates all masks in dataset_processed by:
1. For slices that have a real tumor mask in anon_annotations_nonhidden:
   → resize the original mask to 512x512 (nearest-neighbor) and save
2. For all other slices (unaffected views, no-tumor exams):
   → save an all-black 512x512 mask

This completely replaces whatever ChatGPT's script put there.

Run once — safe to re-run (overwrites existing masks).
"""

import os
import re
import numpy as np
from PIL import Image

# -------------------------------------------------------------------------
# Config — update these if your paths differ
# -------------------------------------------------------------------------
ANNOTATIONS_DIR  = r"C:\Users\culya\Desktop\data_bakalarka\anon_annotations_nonhidden"
PROCESSED_ROOT   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed"
TARGET_SIZE      = (512, 512)   # (width, height) for PIL
SPLITS           = ["train", "val", "test"]
# -------------------------------------------------------------------------


def parse_mask_filename(filename):
    """
    Parse: 00075_20990909_L_CC_1_mask.png
    Returns (patient_id, date, view, exam_idx) or None.
    """
    base = os.path.splitext(filename)[0]
    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)_mask$', base)
    if m:
        return (
            m.group(1),                        # patient_id
            m.group(2),                        # date
            f"{m.group(3)}_{m.group(4)}",      # view  e.g. L_CC
            int(m.group(5)),                   # exam_idx
        )
    return None


def build_annotation_lookup(annotations_dir):
    """
    Returns dict: (patient_id, view, exam_idx) -> full path to original mask file
    """
    lookup = {}
    for fname in os.listdir(annotations_dir):
        if not fname.endswith('.png'):
            continue
        parsed = parse_mask_filename(fname)
        if parsed is None:
            continue
        patient_id, date, view, exam_idx = parsed
        key = (patient_id, view, exam_idx)
        lookup[key] = os.path.join(annotations_dir, fname)
    print(f"Loaded {len(lookup)} original tumor masks from annotations folder.")
    return lookup


def get_image_filenames(images_dir, view):
    """
    Returns list of (exam_idx, filename) sorted by exam_idx
    for a given view folder inside a patient's images dir.
    """
    view_path = os.path.join(images_dir, view)
    if not os.path.isdir(view_path):
        return []

    results = []
    for fname in os.listdir(view_path):
        if not fname.endswith('.png'):
            continue
        base = os.path.splitext(fname)[0]
        # Format: 00075_20990909_L_CC_1
        m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)$', base)
        if m:
            results.append((int(m.group(5)), fname))

    results.sort(key=lambda x: x[0])
    return results


def main():
    annotation_lookup = build_annotation_lookup(ANNOTATIONS_DIR)

    black_mask = Image.fromarray(
        np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8), mode='L'
    )

    total_written  = 0
    total_tumor    = 0
    total_black    = 0
    total_skipped  = 0

    for split in SPLITS:
        for label in ["cancerous", "cancer_free"]:
            label_path = os.path.join(PROCESSED_ROOT, split, label)
            if not os.path.isdir(label_path):
                continue

            for patient in sorted(os.listdir(label_path)):
                patient_path = os.path.join(label_path, patient)
                if not os.path.isdir(patient_path):
                    continue

                images_root = os.path.join(patient_path, 'images')
                masks_root  = os.path.join(patient_path, 'masks')

                if not os.path.isdir(images_root):
                    continue

                # Iterate over all views present in the images folder
                for view in os.listdir(images_root):
                    view_img_path  = os.path.join(images_root, view)
                    view_mask_path = os.path.join(masks_root, view)

                    if not os.path.isdir(view_img_path):
                        continue

                    os.makedirs(view_mask_path, exist_ok=True)

                    exam_list = get_image_filenames(images_root, view)

                    for exam_idx, fname in exam_list:
                        out_mask_path = os.path.join(view_mask_path, fname)

                        # Check if there's a real tumor mask for this slice
                        key = (patient, view, exam_idx)
                        if key in annotation_lookup:
                            # Resize original mask to 512x512 nearest-neighbor
                            orig_mask = Image.open(
                                annotation_lookup[key]
                            ).convert('L')
                            resized = orig_mask.resize(
                                TARGET_SIZE, Image.NEAREST
                            )
                            # Binarize — ensure it's strictly 0 or 255
                            arr = np.array(resized)
                            arr = (arr > 0).astype(np.uint8) * 255
                            Image.fromarray(arr, mode='L').save(out_mask_path)
                            total_tumor += 1
                        else:
                            # No tumor annotation → save black mask
                            black_mask.save(out_mask_path)
                            total_black += 1

                        total_written += 1

    print(f"\nDone.")
    print(f"  Total masks written:      {total_written}")
    print(f"  Tumor masks (resized):    {total_tumor}")
    print(f"  Black masks (no tumor):   {total_black}")
    print(f"  Skipped (no image dir):   {total_skipped}")


if __name__ == "__main__":
    main()