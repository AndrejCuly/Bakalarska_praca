'''
Generator prazdnych masiek pre snimky bez rakoviny.
Je mozne ho pouzit aj pre cele mamogramy aj pre patche,
treba ale upravit sirku a vysku v akom sa vygeneruje maska.
'''

import os
import numpy as np
import cv2

ROOT = r"C:\Example\of\path\to\directory"

HEIGHT = 512
WIDTH = 512

for root, dirs, files in os.walk(ROOT):
    if "images" in root:
        for view in os.listdir(root):
            image_view_path = os.path.join(root, view)
            if not os.path.isdir(image_view_path):
                continue

            mask_view_path = image_view_path.replace("images", "masks")
            os.makedirs(mask_view_path, exist_ok=True)
            for file in os.listdir(image_view_path):
                if not file.endswith(".png"):
                    continue

                image_path = os.path.join(image_view_path, file)
                mask_path = os.path.join(mask_view_path, file)
                mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                cv2.imwrite(mask_path, mask)
                print("Created mask:", mask_path)
print("Done.")
