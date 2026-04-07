"""
dataset3d_clf.py

Per-patient classification dataset for model1_experiment.
Uses rectangular 384x512 images from dataset_processed_v2.
No .pt cache support — loads PNGs directly.
"""

import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

MAX_T    = 6
IMG_H    = 512
IMG_W    = 384
VIEWS    = ['L_CC', 'R_CC', 'L_MLO', 'R_MLO']


def parse_filename(filename):
    base = os.path.splitext(filename)[0]
    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)$', base)
    if m:
        return {'exam_idx': int(m.group(5))}
    return None


class MammogramClassificationDataset(Dataset):
    """
    Per-patient classification dataset.

    Output per sample:
        views:    [4, 1, T, H, W]  float32, normalized [-1, 1]
        pad_mask: [4, T]           bool
        label:    scalar float32   1.0 = cancerous, 0.0 = cancer_free
        patient:  str
    """

    def __init__(
        self,
        cancerous_dir,
        cancer_free_dir=None,
        max_t=MAX_T,
        img_h=IMG_H,
        img_w=IMG_W,
    ):
        self.max_t = max_t
        self.img_h = img_h
        self.img_w = img_w

        self.img_transform = transforms.Compose([
            transforms.Resize((img_h, img_w),
                              interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.samples = []

        if cancerous_dir and os.path.isdir(cancerous_dir):
            self._index_dir(cancerous_dir, label=1)

        if cancer_free_dir and os.path.isdir(cancer_free_dir):
            self._index_dir(cancer_free_dir, label=0)

    def _index_dir(self, root, label):
        for patient in sorted(os.listdir(root)):
            patient_path = os.path.join(root, patient)
            if not os.path.isdir(patient_path):
                continue

            images_root = os.path.join(patient_path, 'images')
            if not os.path.isdir(images_root):
                continue

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

            if all(len(v) == 0 for v in view_files.values()):
                continue

            self.samples.append({
                'label':      label,
                'patient':    patient,
                'view_files': view_files,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s          = self.samples[idx]
        view_files = s['view_files']

        all_views     = []
        all_pad_masks = []

        for view in VIEWS:
            files = view_files[view]
            T     = min(max(len(files) - 1, 1), self.max_t) if len(files) > 0 else 0

            imgs     = []
            pad_mask = []

            for i in range(self.max_t):
                if i < T:
                    img = Image.open(files[i]).convert('L')
                    img = self.img_transform(img)
                    imgs.append(img)
                    pad_mask.append(1)
                else:
                    imgs.append(torch.zeros(1, self.img_h, self.img_w))
                    pad_mask.append(0)

            imgs     = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)
            pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

            all_views.append(imgs)
            all_pad_masks.append(pad_mask)

        views    = torch.stack(all_views,     dim=0)
        pad_mask = torch.stack(all_pad_masks, dim=0)
        label    = torch.tensor(s['label'], dtype=torch.float32)

        return views, pad_mask, label, s['patient']