import os
import cv2
import numpy as np
import random

dataset_root = r"C:\Users\culya\Desktop\data_bakalarka\data\pngs\cancer_free"
output_root = r"C:\Users\culya\Desktop\data_bakalarka\data\patches\crops_cancer_free"

patch_size = 256

VIEWS = ["L_CC", "R_CC", "L_MLO", "R_MLO"]
os.makedirs(output_root, exist_ok=True)

# =========================
# BREAST REGION DETECTION
# =========================

def get_breast_mask(img):
    mask = img > 5
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    breast_mask = (labels == largest).astype(np.uint8)
    kernel = np.ones((25,25), np.uint8)
    breast_mask = cv2.erode(breast_mask, kernel)

    return breast_mask


# Generovanie suradnic pre seriu patchov
def generate_coords(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    idx = random.randint(0, len(xs)-1)
    cx = xs[idx]
    cy = ys[idx]
    half = patch_size // 2
    x = max(0, cx - half)
    y = max(0, cy - half)

    return x, y



patients = os.listdir(dataset_root)
for patient in patients:
    patient_path = os.path.join(dataset_root, patient)
    if not os.path.isdir(patient_path):
        continue

    print("Patient:", patient)
    images_path = os.path.join(patient_path, "images")
    if not os.path.exists(images_path):
        continue

    for view in VIEWS:
        view_path = os.path.join(images_path, view)
        if not os.path.exists(view_path):
            continue

        files = sorted(os.listdir(view_path))
        if len(files) == 0:
            continue

        first_file = files[0]
        if not first_file.endswith(".png"):
            continue

        first_path = os.path.join(view_path, first_file)
        img = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
        breast_mask = get_breast_mask(img)
        coords = generate_coords(breast_mask)
        if coords is None:
            continue

        x, y = coords

        print("  View:", view, "coords:", x, y)

        # crop všetky snímky

        for file in files:
            if not file.endswith(".png"):
                continue

            img_path = os.path.join(view_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            crop = img[y:y+patch_size, x:x+patch_size]
            if crop.shape != (patch_size, patch_size):
                pad_y = patch_size - crop.shape[0]
                pad_x = patch_size - crop.shape[1]

                crop = cv2.copyMakeBorder(
                    crop,
                    0, pad_y,
                    0, pad_x,
                    cv2.BORDER_CONSTANT,
                    value=0
                )

            base = os.path.splitext(file)[0]
            output_path = os.path.join(output_root, base + ".png")

            # prepíše starý patch, takze ked tak zakomentovat!!!
            cv2.imwrite(output_path, crop)

print("\nFinished generating PNG patches.")