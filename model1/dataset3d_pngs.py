import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

MAX_T    = 6
IMG_SIZE = 512   # full mammograms resized to 512x512


def parse_filename(filename):
    """
    Parse a pngs filename into components.
    Format: 00075_20990909_L_CC_1.png
    Returns dict or None if filename doesn't match.
    """
    base = os.path.splitext(filename)[0]
    m = re.match(r'^(\d+)_(\d+)_([LR])_(CC|MLO)_(\d+)$', base)
    if m:
        return {
            'patient_id': m.group(1),
            'date':       m.group(2),
            'laterality': m.group(3),
            'view_type':  m.group(4),
            'view':       f"{m.group(3)}_{m.group(4)}",
            'exam_idx':   int(m.group(5)),
        }
    return None


class MammogramPNGDataset(Dataset):
    """
    Dataset for full mammogram PNGs.

    Each sample is one temporal sequence:
        (patient, view) → T exams stacked along the depth axis

    Folder structure expected:
        root/
          cancerous/
            {patient_id}/
              images/
                L_CC/  R_CC/  L_MLO/  R_MLO/
                  {patient}_{date}_{view}_{exam_idx}.png
              masks/
                L_CC/  R_CC/  L_MLO/  R_MLO/
                  {patient}_{date}_{view}_{exam_idx}.png
          cancer_free/
            {patient_id}/
              images/  masks/  (same structure)

    Output per sample:
        images:   [1, T, H, W]   float32, normalized [-1, 1]
        masks:    [1, T, H, W]   float32, binary {0, 1}
        pad_mask: [T]            bool, True = real slice
        label:    scalar float32, 1.0 = cancerous, 0.0 = cancer_free
    """

    VIEWS = ['L_CC', 'R_CC', 'L_MLO', 'R_MLO']

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

        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        self.samples = []

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
            masks_root  = os.path.join(patient_path, 'masks')

            if not os.path.isdir(images_root):
                continue

            for view in self.VIEWS:
                view_img_path  = os.path.join(images_root, view)
                view_mask_path = os.path.join(masks_root,  view)

                if not os.path.isdir(view_img_path):
                    continue

                # Collect (exam_idx, filename) pairs
                exam_list = []
                for fname in os.listdir(view_img_path):
                    if not fname.endswith('.png'):
                        continue
                    info = parse_filename(fname)
                    if info is None:
                        continue
                    exam_list.append((info['exam_idx'], fname))

                if len(exam_list) == 0:
                    continue

                # Sort by exam index (chronological order)
                exam_list.sort(key=lambda x: x[0])
                filenames = [fname for _, fname in exam_list]

                self.samples.append({
                    'label':      label,
                    'images_dir': view_img_path,
                    'masks_dir':  view_mask_path,
                    'filenames':  filenames,
                })

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        s         = self.samples[idx]
        filenames = s['filenames']
        T         = min(len(filenames), self.max_t)

        images   = []
        masks    = []
        pad_mask = []

        for i in range(self.max_t):
            if i < T:
                fname = filenames[i]

                # Image
                img_path = os.path.join(s['images_dir'], fname)
                img = Image.open(img_path).convert('L')
                img = self.img_transform(img)          # [1, H, W]

                # Mask
                mask_path = os.path.join(s['masks_dir'], fname)
                if os.path.exists(mask_path):
                    mask = Image.open(mask_path).convert('L')
                    mask = self.mask_transform(mask)   # [1, H, W]
                    mask = (mask > 0).float()
                else:
                    mask = torch.zeros(1, self.img_size, self.img_size)

                images.append(img)
                masks.append(mask)
                pad_mask.append(1)
            else:
                images.append(torch.zeros(1, self.img_size, self.img_size))
                masks.append(torch.zeros(1, self.img_size, self.img_size))
                pad_mask.append(0)

        # [T, 1, H, W] → [1, T, H, W]
        images   = torch.stack(images,   dim=0).permute(1, 0, 2, 3)
        masks    = torch.stack(masks,    dim=0).permute(1, 0, 2, 3)
        pad_mask = torch.tensor(pad_mask, dtype=torch.bool)
        label    = torch.tensor(s['label'], dtype=torch.float32)

        return images, masks, pad_mask, label