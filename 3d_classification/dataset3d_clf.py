"""
dataset3d_clf.py

Per-patient classification dataset for the 3D CNN classifier.

Each sample = one patient → tensor of shape [4, 1, T, H, W]
containing all 4 views (L_CC, R_CC, L_MLO, R_MLO) stacked.

Label: 1.0 = cancerous, 0.0 = cancer_free

Missing views (e.g. cancerous patients only have 2 affected views
in the patches dataset) are replaced with zero tensors so the shape
is always [4, 1, T, H, W].

Folder structure expected (same as MammogramPNGDataset):
    root/
      cancerous/
        {patient_id}/
          images/
            L_CC/  R_CC/  L_MLO/  R_MLO/
              {patient}_{date}_{view}_{exam_idx}.png
      cancer_free/
        ...
"""

import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

MAX_T    = 6
IMG_SIZE = 512
VIEWS    = ['L_CC', 'R_CC', 'L_MLO', 'R_MLO']


def parse_filename(filename):
    base = os.path.splitext(filename)[0]
    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)$', base)
    if m:
        return {
            'exam_idx': int(m.group(5)),
        }
    return None


class MammogramClassificationDataset(Dataset):
    """
    Per-patient classification dataset.

    Output per sample:
        views:    [4, 1, T, H, W]  float32, normalized [-1, 1]
                  one tensor per view (L_CC, R_CC, L_MLO, R_MLO)
                  missing views are zero tensors
        pad_mask: [4, T]           bool, True = real slice
        label:    scalar float32,  1.0 = cancerous, 0.0 = cancer_free
        patient:  str              patient ID for logging
    """

    def __init__(
        self,
        cancerous_dir,
        cancer_free_dir=None,
        max_t=MAX_T,
        img_size=IMG_SIZE,
    ):
        self.max_t    = max_t
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.samples = []   # list of dicts, one per patient

        if cancerous_dir and os.path.isdir(cancerous_dir):
            self._index_dir(cancerous_dir, label=1)

        if cancer_free_dir and os.path.isdir(cancer_free_dir):
            self._index_dir(cancer_free_dir, label=0)

    # ------------------------------------------------------------------
    def _index_dir(self, root, label):
        for patient in sorted(os.listdir(root)):
            patient_path = os.path.join(root, patient)
            if not os.path.isdir(patient_path):
                continue

            images_root = os.path.join(patient_path, 'images')
            if not os.path.isdir(images_root):
                continue

            # For each view, collect sorted filenames
            view_files = {}
            for view in VIEWS:
                view_path = os.path.join(images_root, view)
                if not os.path.isdir(view_path):
                    view_files[view] = []
                    continue

                exam_list = []
                for fname in os.listdir(view_path):
                    if not fname.endswith('.png'):
                        continue
                    info = parse_filename(fname)
                    if info is None:
                        continue
                    exam_list.append((info['exam_idx'], fname))

                exam_list.sort(key=lambda x: x[0])
                view_files[view] = [
                    os.path.join(view_path, fname)
                    for _, fname in exam_list
                ]

            # Skip patients with no views at all
            if all(len(v) == 0 for v in view_files.values()):
                continue

            self.samples.append({
                'label':      label,
                'patient':    patient,
                'view_files': view_files,   # dict: view → list of full paths
            })

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        s          = self.samples[idx]
        view_files = s['view_files']

        all_views    = []
        all_pad_masks = []

        for view in VIEWS:
            files = view_files[view]
            T     = min(len(files), self.max_t)

            imgs     = []
            pad_mask = []

            for i in range(self.max_t):
                if i < T:
                    img = Image.open(files[i]).convert('L')
                    img = self.img_transform(img)   # [1, H, W]
                    imgs.append(img)
                    pad_mask.append(1)
                else:
                    imgs.append(torch.zeros(1, self.img_size, self.img_size))
                    pad_mask.append(0)

            # [T, 1, H, W] → [1, T, H, W]
            imgs = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)
            pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

            all_views.append(imgs)
            all_pad_masks.append(pad_mask)

        # Stack views → [4, 1, T, H, W]
        views    = torch.stack(all_views,     dim=0)
        pad_mask = torch.stack(all_pad_masks, dim=0)   # [4, T]
        label    = torch.tensor(s['label'], dtype=torch.float32)

        return views, pad_mask, label, s['patient']