import os
import cv2
import numpy as np
import pydicom

dicom_folder = r"D:\CSAW\2021-204-1-1\data"        # SSD folder with dicoms
mask_folder = r"C:\Users\user\Desktop\binary_masks"          # local masks
output_folder = r"C:\Users\user\Desktop\where\tp\output"

patch_size = 256

os.makedirs(output_folder, exist_ok=True)

def load_dicom(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)

    return img

# BUILD DICOM LOOKUP

print("Scanning DICOM folder...")
dicom_lookup = {}
for file in os.listdir(dicom_folder):
    if file.startswith("._"):
        continue

    if not file.endswith(".dcm"):
        continue

    name = os.path.splitext(file)[0]
    dicom_lookup[name] = os.path.join(dicom_folder, file)

print("Found", len(dicom_lookup), "dicom files")

# ============================
# PROCESS MASKS
# ============================
mask_files = os.listdir(mask_folder)
print("Masks found:", len(mask_files))
for mask_file in mask_files:
    if mask_file.startswith("._"):
        continue

    if not mask_file.lower().endswith(".png"):
        continue

    base_name = os.path.splitext(mask_file)[0]
    if base_name not in dicom_lookup:
        print("Missing DICOM:", base_name)
        continue

    dicom_path = dicom_lookup[base_name]
    mask_path = os.path.join(mask_folder, mask_file)
    print("Processing:", base_name)
    mamm = load_dicom(dicom_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("Mask load failed:", base_name)
        continue

    if mamm.shape != mask.shape:
        print("Size mismatch:", base_name)
        continue

    #hladanie oblasti nadoru
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        print("Empty mask:", base_name)
        continue

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    #center nadoru
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    half = patch_size // 2

    x1 = center_x - half
    x2 = center_x + half
    y1 = center_y - half
    y2 = center_y + half

    # clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(mamm.shape[1], x2)
    y2 = min(mamm.shape[0], y2)

    crop = mamm[y1:y2, x1:x2]

    if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
        crop = cv2.copyMakeBorder(
            crop,
            0,
            patch_size - crop.shape[0],
            0,
            patch_size - crop.shape[1],
            cv2.BORDER_CONSTANT,
            value=0
        )

    output_path = os.path.join(output_folder, base_name + "_tumor.png")
    cv2.imwrite(output_path, crop)

print("\nFinished processing dataset")
