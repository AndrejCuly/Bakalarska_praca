import os
import cv2
import numpy as np
import time

mask_folder = r"C:\Users\culya\Desktop\data_bakalarka\anon_annotations_nonhidden"
dataset_root = r"C:\Users\culya\Desktop\data_bakalarka\data\patches\crops_cancer"

patch_size = 256


# PROCESS MASKS

for root, dirs, files in os.walk(dataset_root):

    if "images" not in root:
        continue

    view = os.path.basename(root)

    patient = root.split(os.sep)[-3]

    mask_view_folder = root.replace("images", "masks")

    os.makedirs(mask_view_folder, exist_ok=True)

    for file in files:

        if not file.endswith(".png"):
            continue

        filename = os.path.splitext(file)[0]

        # odstráni _tumor z názvu patchu
        base_name = filename.replace("_tumor", "")
        mask_name = base_name + "_mask.png"
        original_mask_path = os.path.join(mask_folder, mask_name)

        print("PATCH:", file)
        print("LOOKING FOR:", mask_name)

        if not os.path.exists(original_mask_path):
            print("Missing mask:", mask_name)
            continue

        mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            continue

        # tumor center
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        half = patch_size // 2

        x1 = center_x - half
        x2 = center_x + half
        y1 = center_y - half
        y2 = center_y + half

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)

        crop = mask[y1:y2, x1:x2]

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

        output_path = os.path.join(mask_view_folder, file)

        #prepise stary subor, ked tak zakomentovat
        cv2.imwrite(output_path, crop)
        print("Saved mask:", output_path)


print("\nFinished")