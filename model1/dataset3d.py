import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Maximum number of exams to pad/truncate to (the T dimension)
MAX_T = 6
IMG_SIZE = 256


def parse_filename(filename):
    """
    Parse a patch filename into its components.

    Cancerous:   00075_20990909_L_CC_1_tumor
    Cancer-free plain:   00004_20990909_L_CC_1
    Cancer-free hard neg: 00004_20990909_L_CC_1_normal_0

    Returns dict with keys: patient_id, date, laterality, view_type, exam_idx, patch_type
    patch_type is one of: 'tumor', 'normal_0', 'plain'
    """
    base = os.path.splitext(filename)[0]

    # tumor
    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)_tumor$', base)
    if m:
        return {
            'patient_id': m.group(1),
            'date': m.group(2),
            'laterality': m.group(3),
            'view_type': m.group(4),
            'view': f"{m.group(3)}_{m.group(4)}",
            'exam_idx': int(m.group(5)),
            'patch_type': 'tumor'
        }

    # normal_0
    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)_normal_0$', base)
    if m:
        return {
            'patient_id': m.group(1),
            'date': m.group(2),
            'laterality': m.group(3),
            'view_type': m.group(4),
            'view': f"{m.group(3)}_{m.group(4)}",
            'exam_idx': int(m.group(5)),
            'patch_type': 'normal_0'
        }

    # plain (cancer-free, no suffix)
    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)$', base)
    if m:
        return {
            'patient_id': m.group(1),
            'date': m.group(2),
            'laterality': m.group(3),
            'view_type': m.group(4),
            'view': f"{m.group(3)}_{m.group(4)}",
            'exam_idx': int(m.group(5)),
            'patch_type': 'plain'
        }

    return None


class Mammogram3DDataset(Dataset):
    """
    Each sample is one temporal sequence:
        (patient, view, patch_type) → T exams stacked along the depth axis

    Output shapes:
        images: [1, T, H, W]   (1 grayscale channel, T time steps)
        masks:  [1, T, H, W]

    Padding: sequences shorter than MAX_T are zero-padded at the END.
    A padding mask is also returned so the loss can ignore padded slices.

    Args:
        cancer_dir:      path to the cancerous patients folder
        cancer_free_dir: path to the cancer-free patients folder (optional)
        max_t:           maximum sequence length (pad/truncate to this)
        patch_types:     which patch types to include, subset of
                         {'tumor', 'plain', 'normal_0'}
                         Default: all three.
    """

    def __init__(
        self,
        cancer_dir,
        cancer_free_dir=None,
        max_t=MAX_T,
        patch_types=None,
    ):
        self.max_t = max_t

        if patch_types is None:
            patch_types = {'tumor', 'plain', 'normal_0'}
        self.patch_types = set(patch_types)

        self.img_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        # Each entry: (label, images_dir, masks_dir, view, patch_type, sorted_filenames)
        # label: 1 = cancerous sequence, 0 = cancer-free sequence
        self.samples = []

        if cancer_dir and os.path.isdir(cancer_dir):
            self._index_dir(cancer_dir, label=1)

        if cancer_free_dir and os.path.isdir(cancer_free_dir):
            self._index_dir(cancer_free_dir, label=0)

    # ------------------------------------------------------------------
    def _index_dir(self, root, label):
        for patient in sorted(os.listdir(root)):
            patient_path = os.path.join(root, patient)
            if not os.path.isdir(patient_path):
                continue

            images_root = os.path.join(patient_path, 'images')
            masks_root = os.path.join(patient_path, 'masks')

            if not os.path.isdir(images_root):
                continue

            # Collect all files grouped by (view, patch_type)
            # key: (view_str, patch_type) → list of (exam_idx, filename)
            groups = {}

            for view_dir in sorted(os.listdir(images_root)):
                view_path = os.path.join(images_root, view_dir)
                if not os.path.isdir(view_path):
                    continue

                for fname in os.listdir(view_path):
                    if not fname.endswith('.png'):
                        continue
                    info = parse_filename(fname)
                    if info is None:
                        continue
                    if info['patch_type'] not in self.patch_types:
                        continue

                    key = (info['view'], info['patch_type'])
                    groups.setdefault(key, []).append((info['exam_idx'], fname))

            # Each group becomes one sample (temporal sequence)
            for (view, patch_type), exam_list in groups.items():
                # Sort by exam index
                exam_list.sort(key=lambda x: x[0])

                images_dir = os.path.join(images_root, view)
                masks_dir = os.path.join(masks_root, view)

                filenames = [fname for _, fname in exam_list]

                self.samples.append({
                    'label': label,
                    'images_dir': images_dir,
                    'masks_dir': masks_dir,
                    'filenames': filenames,   # ordered by exam index
                })

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        s = self.samples[idx]
        filenames = s['filenames']
        T = min(len(filenames), self.max_t)

        images = []
        masks = []
        pad_mask = []   # 1 = real slice, 0 = padded

        for i in range(self.max_t):
            if i < T:
                fname = filenames[i]

                img_path = os.path.join(s['images_dir'], fname)
                img = Image.open(img_path).convert('L')
                img = self.img_transform(img)          # [1, H, W]

                mask_path = os.path.join(s['masks_dir'], fname)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask = self.mask_transform(mask)   # [1, H, W]
                    mask = (mask > 0).float()
                else:
                    # Image exists but mask doesn't → all-black mask
                    mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)

                images.append(img)
                masks.append(mask)
                pad_mask.append(1)
            else:
                # Padding slice
                images.append(torch.zeros(1, IMG_SIZE, IMG_SIZE))
                masks.append(torch.zeros(1, IMG_SIZE, IMG_SIZE))
                pad_mask.append(0)

        # Stack along time dimension → [T, 1, H, W] → permute → [1, T, H, W]
        images = torch.stack(images, dim=0).permute(1, 0, 2, 3)   # [1, T, H, W]
        masks = torch.stack(masks, dim=0).permute(1, 0, 2, 3)     # [1, T, H, W]
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool)        # [T]

        label = torch.tensor(s['label'], dtype=torch.float32)

        return images, masks, pad_mask, label