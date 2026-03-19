"""
dataset3d_clf_patches.py

Patch-level classification dataset for the 3D CNN classifier.

Label logic:
    _tumor patches from cancerous patients    → label 1
    _normal_0 patches from cancerous patients → label 0
    all cancer_free patient patches           → label 0

Each sample = one (patient, view, patch_type) temporal sequence
→ tensor of shape [4, 1, T, H, W] with all 4 views stacked.

Missing views are replaced with zero tensors.

Folder structure expected:
    root/
      cancer/
        {patient_id}/
          images/
            L_CC/  R_CC/  L_MLO/  R_MLO/
              {patient}_{date}_{view}_{examidx}_tumor.png
              {patient}_{date}_{view}_{examidx}_normal_0.png  (maybe)
      cancer_free/
        {patient_id}/
          images/
            L_CC/  R_CC/  L_MLO/  R_MLO/
              {patient}_{date}_{view}_{examidx}.png
              {patient}_{date}_{view}_{examidx}_normal_0.png
"""

import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

MAX_T    = 6
IMG_SIZE = 256
VIEWS    = ['L_CC', 'R_CC', 'L_MLO', 'R_MLO']


def parse_patch_filename(filename):
    """
    Parse patch filename into components.

    Patterns:
        00075_20990909_L_CC_1_tumor    → patch_type='tumor'
        00004_20990909_L_CC_1_normal_0 → patch_type='normal_0'
        00004_20990909_L_CC_1          → patch_type='plain'

    Returns dict or None.
    """
    base = os.path.splitext(filename)[0]

    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)_tumor$', base)
    if m:
        return {
            'patient_id': m.group(1),
            'view':       f"{m.group(3)}_{m.group(4)}",
            'exam_idx':   int(m.group(5)),
            'patch_type': 'tumor',
        }

    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)_normal_0$', base)
    if m:
        return {
            'patient_id': m.group(1),
            'view':       f"{m.group(3)}_{m.group(4)}",
            'exam_idx':   int(m.group(5)),
            'patch_type': 'normal_0',
        }

    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)$', base)
    if m:
        return {
            'patient_id': m.group(1),
            'view':       f"{m.group(3)}_{m.group(4)}",
            'exam_idx':   int(m.group(5)),
            'patch_type': 'plain',
        }

    return None


class PatchClassificationDataset(Dataset):
    """
    Patch-level classification dataset.

    Each sample = one (patient, view, patch_type) temporal sequence.

    Output per sample:
        views:    [4, 1, T, H, W]  float32, normalized [-1, 1]
        pad_mask: [4, T]           bool
        label:    scalar float32   1.0 = tumor patch, 0.0 = normal patch
        patient:  str
    """

    def __init__(
        self,
        cancer_dir,
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

        self.samples = []

        if cancer_dir and os.path.isdir(cancer_dir):
            self._index_cancer_dir(cancer_dir)

        if cancer_free_dir and os.path.isdir(cancer_free_dir):
            self._index_cancer_free_dir(cancer_free_dir)

    # ------------------------------------------------------------------
    def _index_cancer_dir(self, root):
        """
        Cancer patients: _tumor → label 1, _normal_0 → label 0
        Group by (patient, patch_type) across all views.
        """
        for patient in sorted(os.listdir(root)):
            patient_path = os.path.join(root, patient)
            if not os.path.isdir(patient_path):
                continue

            images_root = os.path.join(patient_path, 'images')
            if not os.path.isdir(images_root):
                continue

            # Group files by patch_type
            # key: patch_type → view → list of (exam_idx, full_path)
            groups = {'tumor': {}, 'normal_0': {}}

            for view in VIEWS:
                view_path = os.path.join(images_root, view)
                if not os.path.isdir(view_path):
                    continue

                for fname in os.listdir(view_path):
                    if not fname.endswith('.png'):
                        continue
                    info = parse_patch_filename(fname)
                    if info is None:
                        continue
                    pt = info['patch_type']
                    if pt not in groups:
                        continue
                    if view not in groups[pt]:
                        groups[pt][view] = []
                    groups[pt][view].append(
                        (info['exam_idx'],
                         os.path.join(view_path, fname))
                    )

            # Create one sample per patch_type that has at least one view
            for patch_type, view_dict in groups.items():
                if all(len(v) == 0 for v in view_dict.values()):
                    continue

                label = 1 if patch_type == 'tumor' else 0

                # Sort each view by exam index
                view_files = {}
                for view in VIEWS:
                    files = view_dict.get(view, [])
                    files.sort(key=lambda x: x[0])
                    view_files[view] = [f for _, f in files]

                self.samples.append({
                    'label':      label,
                    'patient':    patient,
                    'patch_type': patch_type,
                    'view_files': view_files,
                })

    # ------------------------------------------------------------------
    def _index_cancer_free_dir(self, root):
        """
        Cancer-free patients: all patches → label 0
        Group by (patient, patch_type) across all views.
        """
        for patient in sorted(os.listdir(root)):
            patient_path = os.path.join(root, patient)
            if not os.path.isdir(patient_path):
                continue

            images_root = os.path.join(patient_path, 'images')
            if not os.path.isdir(images_root):
                continue

            groups = {'plain': {}, 'normal_0': {}}

            for view in VIEWS:
                view_path = os.path.join(images_root, view)
                if not os.path.isdir(view_path):
                    continue

                for fname in os.listdir(view_path):
                    if not fname.endswith('.png'):
                        continue
                    info = parse_patch_filename(fname)
                    if info is None:
                        continue
                    pt = info['patch_type']
                    if pt not in groups:
                        continue
                    if view not in groups[pt]:
                        groups[pt][view] = []
                    groups[pt][view].append(
                        (info['exam_idx'],
                         os.path.join(view_path, fname))
                    )

            for patch_type, view_dict in groups.items():
                if all(len(v) == 0 for v in view_dict.values()):
                    continue

                view_files = {}
                for view in VIEWS:
                    files = view_dict.get(view, [])
                    files.sort(key=lambda x: x[0])
                    view_files[view] = [f for _, f in files]

                self.samples.append({
                    'label':      0,
                    'patient':    patient,
                    'patch_type': patch_type,
                    'view_files': view_files,
                })

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        s          = self.samples[idx]
        view_files = s['view_files']

        all_views     = []
        all_pad_masks = []

        for view in VIEWS:
            files = view_files.get(view, [])
            T     = min(len(files), self.max_t)

            imgs     = []
            pad_mask = []

            for i in range(self.max_t):
                if i < T:
                    img = Image.open(files[i]).convert('L')
                    img = self.img_transform(img)
                    imgs.append(img)
                    pad_mask.append(1)
                else:
                    imgs.append(torch.zeros(1, self.img_size, self.img_size))
                    pad_mask.append(0)

            imgs     = torch.stack(imgs, dim=0).permute(1, 0, 2, 3)
            pad_mask = torch.tensor(pad_mask, dtype=torch.bool)

            all_views.append(imgs)
            all_pad_masks.append(pad_mask)

        views    = torch.stack(all_views,     dim=0)   # [4, 1, T, H, W]
        pad_mask = torch.stack(all_pad_masks, dim=0)   # [4, T]
        label    = torch.tensor(s['label'], dtype=torch.float32)

        return views, pad_mask, label, s['patient']