import os
import cv2
import numpy as np

INPUT_ROOT = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_split"
OUTPUT_ROOT = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed"

TARGET_SIZE = (512, 512)

def remove_black_borders(img):
    """Crop the largest non-black region."""
    thresh = img > 5
    coords = np.argwhere(thresh)

    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img[y0:y1, x0:x1]

for split in ["train", "val", "test"]:
    for cls in ["cancerous", "cancer_free"]:

        split_path = os.path.join(INPUT_ROOT, split, cls)

        if not os.path.exists(split_path):
            continue

        for patient in os.listdir(split_path):

            patient_path = os.path.join(split_path, patient)

            img_root = os.path.join(patient_path, "images")
            mask_root = os.path.join(patient_path, "masks")

            for view in os.listdir(img_root):

                img_view_path = os.path.join(img_root, view)
                mask_view_path = os.path.join(mask_root, view)

                for file in os.listdir(img_view_path):

                    img_path = os.path.join(img_view_path, file)
                    mask_path = os.path.join(mask_view_path, file)

                    img = cv2.imread(img_path, 0)
                    mask = cv2.imread(mask_path, 0)

                    if img is None or mask is None:
                        print("Missing file:", img_path)
                        continue

                    # remove borders
                    img = remove_black_borders(img)
                    mask = remove_black_borders(mask)

                    # resize
                    img = cv2.resize(img, TARGET_SIZE)
                    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

                    # normalize image
                    img = img.astype(np.float32) / 255.0

                    # convert mask to binary
                    mask = (mask > 0).astype(np.uint8)

                    # output path
                    out_img_dir = os.path.join(OUTPUT_ROOT, split, cls, patient, "images", view)
                    out_mask_dir = os.path.join(OUTPUT_ROOT, split, cls, patient, "masks", view)

                    os.makedirs(out_img_dir, exist_ok=True)
                    os.makedirs(out_mask_dir, exist_ok=True)

                    out_img_path = os.path.join(out_img_dir, file)
                    out_mask_path = os.path.join(out_mask_dir, file)

                    cv2.imwrite(out_img_path, (img * 255).astype(np.uint8))
                    cv2.imwrite(out_mask_path, mask * 255)

                    print("Processed:", out_img_path)

print("Preprocessing finished.")