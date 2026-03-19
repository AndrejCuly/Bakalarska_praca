import os
import numpy as np
from PIL import Image

root = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed"

for split in ['train', 'val', 'test']:
    path = os.path.join(root, split, 'cancerous')
    if not os.path.isdir(path):
        continue
    for patient in sorted(os.listdir(path)):
        masks_root = os.path.join(path, patient, 'masks')
        if not os.path.isdir(masks_root):
            continue
        found = False
        for view in os.listdir(masks_root):
            view_path = os.path.join(masks_root, view)
            if not os.path.isdir(view_path):
                continue
            for f in os.listdir(view_path):
                if not f.endswith('.png'):
                    continue
                m = np.array(Image.open(os.path.join(view_path, f)))
                if m.max() > 0:
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"{split}  {patient}")