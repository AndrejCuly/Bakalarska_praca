"""
gradcam.py

3D GradCAM visualization for mammogram sequence classifiers.
Generates heatmap overlays for all 4 views of true positive patients.

Works with:
    - Model 1 (MammogramClassifier         — custom 3D CNN)
    - Model 2 (MammogramResNetClassifier   — MedicalNet ResNet-10)
    - Model 3 (MammogramVNetClassifier     — VNet)
    - Model 4 (MammogramResNet50Classifier — ResNet-50 3D)

For each true positive patient, for each view:
    - Saves a SEPARATE figure per view: {patient_id}_{view}_gradcam.png
    - Each figure has one row per real exam
    - Each row shows 3 panels: Original | GradCAM overlay | Mask

Usage:
    Edit CONFIG section below, then run:
    python gradcam.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# CONFIG — edit these for each run
# -------------------------------------------------------------------------

MODEL_TYPE = "model4"   # "model1" | "model2" | "model3" | "model4"
DATA_TYPE  = "patches"    # "whole" | "whole_v2" (512x384) | "patches"

CANCEROUS_TEST_WHOLE      = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancerous"
CANCER_FREE_TEST_WHOLE    = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed\test\cancer_free"

CANCEROUS_TEST_WHOLE_V2   = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed_v2\test\cancerous"
CANCER_FREE_TEST_WHOLE_V2 = r"C:\Users\culya\Desktop\data_bakalarka\data\dataset_processed_v2\test\cancer_free"

CANCER_TEST_PATCHES      = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\test\cancer"
CANCER_FREE_TEST_PATCHES = r"C:\Users\culya\Desktop\data_bakalarka\data\patches_split\test\cancer_free"

CHECKPOINT_MODEL1_WHOLE    = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model1\experiments\classification\whole_mamms\results\best_model_clf.pth"
CHECKPOINT_MODEL1_WHOLE_V2 = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model1_experiment\results\best_model_clf.pth"
CHECKPOINT_MODEL1_PATCHES  = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model1\experiments\classification\patches\results\best_model_clf_patches.pth"
CHECKPOINT_MODEL2_WHOLE    = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model2\experiments\classification\whole_mamms\results\best_model_resnet_clf.pth"
CHECKPOINT_MODEL2_PATCHES  = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model2\experiments\classification\patches\results\best_model_resnet_clf_patches.pth"
CHECKPOINT_MODEL3_WHOLE    = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model3\experiments\classification\whole_mamms\results\best_model_vnet_clf.pth"
CHECKPOINT_MODEL3_PATCHES  = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model3\experiments\classification\patches\results\best_model_vnet_clf_patches.pth"
CHECKPOINT_MODEL4_WHOLE    = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model4\experiments\classification\whole_mamms\results\best_model_resnet50_clf.pth"
CHECKPOINT_MODEL4_PATCHES  = r"C:\Skola\Bakalarka\Model1\default\pngs_processed\model4\experiments\classification\patches\results\best_model_resnet50_clf_patches.pth"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'gradcam',
                          f'{MODEL_TYPE}_{DATA_TYPE}')

# -------------------------------------------------------------------------
# Dynamic imports
# -------------------------------------------------------------------------

if MODEL_TYPE == "model1":
    from model1.experiments.classification.classifier3d import MammogramClassifier
elif MODEL_TYPE == "model2":
    from model2.resnet3d import MammogramResNetClassifier
elif MODEL_TYPE == "model3":
    from model3.vnet3d import MammogramVNetClassifier
elif MODEL_TYPE == "model4":
    from model4.resnet50_3d import MammogramResNet50Classifier

if DATA_TYPE in ("whole", "whole_v2"):
    from model1.experiments.classification.whole_mamms.dataset3d_clf import MammogramClassificationDataset
else:
    from model1.experiments.classification.patches.dataset3d_clf_patches import PatchClassificationDataset

# -------------------------------------------------------------------------
# GradCAM implementation
# -------------------------------------------------------------------------

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self.forward_hook  = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """
        Generate GradCAM heatmap for input_tensor [B, 1, T, H, W].
        Returns cam [T, H_cam, W_cam] normalized per-slice.
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0].backward()

        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.ndim == 2:
            cam = cam[np.newaxis, ...]

        # Per-slice percentile stretch
        for t in range(cam.shape[0]):
            s = cam[t]
            p_low  = np.percentile(s, 50)
            p_high = np.percentile(s, 99)
            if p_high > p_low:
                cam[t] = np.clip((s - p_low) / (p_high - p_low), 0, 1)

        return cam

    def generate_per_slice(self, input_tensor, n_real):
        """
        For models where temporal pooling collapses T early (Model 2, Model 4),
        run GradCAM independently for each exam slice.
        input_tensor: [B, 1, T, H, W]
        Returns cam [n_real, H_cam, W_cam].
        """
        slices = []
        for t in range(n_real):
            single_slice = input_tensor[:, :, t:t+1, :, :]
            single_slice = single_slice.clone().requires_grad_(True)

            self.model.eval()
            output = self.model(single_slice)
            self.model.zero_grad()
            output[0].backward()

            weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)
            cam_t = (weights * self.activations).sum(dim=1, keepdim=True)
            cam_t = torch.relu(cam_t).squeeze().cpu().numpy()

            if cam_t.ndim == 3:
                cam_t = cam_t[0]

            p_low  = np.percentile(cam_t, 50)
            p_high = np.percentile(cam_t, 99)
            if p_high > p_low:
                cam_t = np.clip((cam_t - p_low) / (p_high - p_low), 0, 1)

            slices.append(cam_t)

        return np.stack(slices, axis=0)

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def get_target_layer(model, model_type):
    """
    Return the last convolutional layer before GAP for each model.

    Model 1 — encoder.block4.conv2
        Last Conv3d in the 4th ConvBlock of the UNet encoder.

    Model 2 — encoder.layer4[0].conv2
        Last Conv3d of the first (only) BasicBlock in ResNet-10 layer4.
        Uses generate_per_slice because ResNet-10 has a large initial stride
        that collapses spatial dims quickly; per-slice gives cleaner maps.

    Model 3 — encoder.down_tr256.ops[-1].conv1
        Last LUConv inside the final DownTransition block of the VNet encoder.
        down_tr256 doubles channels to 256; ops is an nn.Sequential of LUConv
        modules; [-1] is the last one; .conv1 is its Conv3d.

    Model 4 — encoder.layer4[2].conv3
        Last Conv3d of the last Bottleneck3D in ResNet-50 layer4 (3 blocks,
        index 2). conv3 is the 1×1×1 expansion conv, the final conv before
        residual add and ReLU.
        Uses generate_per_slice for the same reason as Model 2.
    """
    if model_type == "model1":
        return model.encoder.block4.conv2
    elif model_type == "model2":
        return model.encoder.layer4[0].conv2
    elif model_type == "model3":
        return model.encoder.down_tr256.ops[-1].conv1
    elif model_type == "model4":
        return model.encoder.layer4[2].conv3
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Models where GAP collapses T — use per-slice GradCAM
PER_SLICE_MODELS = {"model2", "model4"}


class SingleViewEncoder(torch.nn.Module):
    """Wraps the full model to process a single view [B, 1, T, H, W]."""
    def __init__(self, full_model):
        super().__init__()
        self.encoder    = full_model.encoder
        self.classifier = full_model.classifier

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)


# -------------------------------------------------------------------------
# Mask loading
# -------------------------------------------------------------------------

def load_masks_for_patient(patient_id, cancerous_dir, view_names):
    patient_path = os.path.join(cancerous_dir, patient_id)
    masks_root   = os.path.join(patient_path, 'masks')
    result = {}
    for view in view_names:
        view_mask_dir = os.path.join(masks_root, view)
        if not os.path.isdir(view_mask_dir):
            result[view] = []
            continue
        mask_files = sorted([f for f in os.listdir(view_mask_dir) if f.endswith('.png')])
        masks = []
        for mf in mask_files:
            m = cv2.imread(os.path.join(view_mask_dir, mf), cv2.IMREAD_GRAYSCALE)
            masks.append(m)
        result[view] = masks
    return result

def load_masks_for_patient_patches(patient_id, cancerous_dir, view_names):
    patient_path = os.path.join(cancerous_dir, patient_id)
    masks_root   = os.path.join(patient_path, 'masks')
    result = {}
    for view in view_names:
        view_mask_dir = os.path.join(masks_root, view)
        if not os.path.isdir(view_mask_dir):
            result[view] = []
            continue
        mask_files = sorted([f for f in os.listdir(view_mask_dir) if f.endswith('.png')])
        masks = []
        for mf in mask_files:
            m = cv2.imread(os.path.join(view_mask_dir, mf), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                masks.append(m)
        result[view] = masks
    return result


# -------------------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------------------

def apply_heatmap(image_np, cam_slice, alpha=0.65):
    """Blend a CAM slice onto a grayscale image, return RGB uint8."""
    h, w = image_np.shape
    cam_resized = cv2.resize(cam_slice.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_LINEAR)
    cam_resized = cv2.GaussianBlur(cam_resized, (15, 15), 0)
    cam_resized = np.clip(cam_resized, 0, 1)

    img_rgb   = np.stack([image_np] * 3, axis=-1)
    img_uint8 = (img_rgb * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = ((1 - alpha) * img_uint8 + alpha * heatmap).astype(np.uint8)
    return overlay


def mask_to_display(mask_np, image_shape):
    """
    Convert a binary mask to an RGB display image.
    Tumor region → bright red; background → dark gray.
    Draws a yellow crosshair at the centroid if tumor present.
    """
    h, w = image_shape
    mask_resized = cv2.resize(mask_np.astype(np.float32), (w, h),
                              interpolation=cv2.INTER_NEAREST)
    has_tumor = mask_resized.max() > 127

    display = np.zeros((h, w, 3), dtype=np.uint8)
    display[:] = (30, 30, 30)

    if has_tumor:
        tumor_px = mask_resized > 127
        display[tumor_px] = (220, 30, 30)
        coords = np.argwhere(tumor_px)
        cy = int(coords[:, 0].mean())
        cx = int(coords[:, 1].mean())
        cv2.drawMarker(display, (cx, cy), (255, 255, 0),
                       cv2.MARKER_CROSS, markerSize=40, thickness=2)
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "No tumor"
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        cv2.putText(display, text,
                    ((w - tw) // 2, (h + th) // 2),
                    font, 0.7, (150, 150, 150), 2, cv2.LINE_AA)

    return display


LABEL_MAP = {
    'L_CC':  'Left CC',
    'R_CC':  'Right CC',
    'L_MLO': 'Left MLO',
    'R_MLO': 'Right MLO',
}

COLS = ['Original', 'GradCAM Overlay', 'Mask']


def save_gradcam_per_view(views_tensor, pad_mask_tensor, cam_dict,
                          patient_id, masks_dict, output_dir,
                          prob, view_names):
    """
    Save one figure per view.
    Rows = real exams; cols = [Original | GradCAM | Mask].
    """
    real_t_per_view = pad_mask_tensor.sum(dim=1).tolist()
    os.makedirs(output_dir, exist_ok=True)

    for vi, view_name in enumerate(view_names):
        n_real     = int(real_t_per_view[vi])
        cam        = cam_dict.get(view_name)
        view_masks = masks_dict.get(view_name, [])

        if n_real == 0:
            print(f"  {view_name}: no real exams, skipping.")
            continue

        panel_w = 5.5
        panel_h = 6.5
        fig_w = panel_w * 3 + 1.0
        fig_h = panel_h * n_real + 1.2

        fig, axes = plt.subplots(n_real, 3,
                                 figsize=(fig_w, fig_h),
                                 squeeze=False)

        for ti in range(n_real):
            img_slice = views_tensor[vi, 0, ti].cpu().numpy()
            img_slice = np.clip((img_slice + 1) / 2, 0, 1)
            h_img, w_img = img_slice.shape

            # --- Col 0: Original ---
            axes[ti, 0].imshow(img_slice, cmap='gray', vmin=0, vmax=1)

            # --- Col 1: GradCAM overlay ---
            if cam is not None and cam.ndim == 3 and ti < cam.shape[0]:
                overlay = apply_heatmap(img_slice, cam[ti])
                axes[ti, 1].imshow(overlay)
            else:
                axes[ti, 1].imshow(img_slice, cmap='gray', vmin=0, vmax=1)
                axes[ti, 1].text(0.5, 0.5, 'No CAM',
                                 transform=axes[ti, 1].transAxes,
                                 ha='center', va='center', fontsize=13,
                                 color='white', fontweight='bold')

            # Tumor crosshair on Original + Overlay
            if ti < len(view_masks) and view_masks[ti] is not None:
                mask = view_masks[ti]
                coords = np.argwhere(mask > 127)
                if len(coords) > 0:
                    cy = coords[:, 0].mean()
                    cx = coords[:, 1].mean()
                    mask_h, mask_w = mask.shape
                    cx_s = np.clip(cx * w_img / mask_w, 0, w_img - 1)
                    cy_s = np.clip(cy * h_img / mask_h, 0, h_img - 1)
                    for ax_col in [0, 1]:
                        axes[ti, ax_col].plot(cx_s, cy_s, 'r+',
                                              markersize=22, markeredgewidth=2.5)
                        axes[ti, ax_col].plot(cx_s, cy_s, 'ro',
                                              markersize=10,
                                              markerfacecolor='none',
                                              markeredgewidth=2,
                                              markeredgecolor='red')

            # --- Col 2: Mask ---
            if ti < len(view_masks) and view_masks[ti] is not None:
                mask_display = mask_to_display(view_masks[ti], (h_img, w_img))
                axes[ti, 2].imshow(mask_display)
            else:
                axes[ti, 2].imshow(np.zeros((h_img, w_img, 3), dtype=np.uint8) + 30)
                axes[ti, 2].text(0.5, 0.5, 'No mask',
                                 transform=axes[ti, 2].transAxes,
                                 ha='center', va='center', fontsize=13,
                                 color='gray', fontweight='bold')

            axes[ti, 0].set_ylabel(f'Exam {ti + 1}', fontsize=13,
                                   fontweight='bold', rotation=90,
                                   labelpad=8, va='center')

            if ti == 0:
                for ci, col_title in enumerate(COLS):
                    axes[ti, ci].set_title(col_title, fontsize=13,
                                           fontweight='bold', pad=8)

            for ci in range(3):
                axes[ti, ci].set_xticks([])
                axes[ti, ci].set_yticks([])
                for spine in axes[ti, ci].spines.values():
                    spine.set_visible(False)

        red_patch    = mpatches.Patch(color='red',    label='Tumor region (mask)')
        yellow_patch = mpatches.Patch(color='yellow', label='Tumor centroid')
        fig.legend(handles=[red_patch, yellow_patch],
                   loc='lower right', fontsize=10, framealpha=0.85,
                   bbox_to_anchor=(0.99, 0.01))

        view_label = LABEL_MAP.get(view_name, view_name)
        title = (f'GradCAM — Patient {patient_id} | {view_label} | '
                 f'{MODEL_TYPE.upper()} | {DATA_TYPE} | '
                 f'Cancer prob: {prob:.3f}')
        plt.suptitle(title, fontsize=13, fontweight='bold', y=1.0)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        save_path = os.path.join(output_dir, f'{patient_id}_{view_name}_gradcam.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")


# -------------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------------

def build_model(model_type, embed_dim=256):
    if model_type == "model1":
        return MammogramClassifier(num_views=4, embed_dim=embed_dim)
    elif model_type == "model2":
        return MammogramResNetClassifier(num_views=4, embed_dim=embed_dim)
    elif model_type == "model3":
        return MammogramVNetClassifier(num_views=4, embed_dim=embed_dim)
    elif model_type == "model4":
        return MammogramResNet50Classifier(num_views=4, embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_checkpoint(model_type, data_type):
    mapping = {
        ("model1", "whole"):    CHECKPOINT_MODEL1_WHOLE,
        ("model1", "whole_v2"): CHECKPOINT_MODEL1_WHOLE_V2,
        ("model1", "patches"):  CHECKPOINT_MODEL1_PATCHES,
        ("model2", "whole"):    CHECKPOINT_MODEL2_WHOLE,
        ("model2", "whole_v2"): CHECKPOINT_MODEL2_WHOLE,
        ("model2", "patches"):  CHECKPOINT_MODEL2_PATCHES,
        ("model3", "whole"):    CHECKPOINT_MODEL3_WHOLE,
        ("model3", "whole_v2"): CHECKPOINT_MODEL3_WHOLE,
        ("model3", "patches"):  CHECKPOINT_MODEL3_PATCHES,
        ("model4", "whole"):    CHECKPOINT_MODEL4_WHOLE,
        ("model4", "whole_v2"): CHECKPOINT_MODEL4_WHOLE,
        ("model4", "patches"):  CHECKPOINT_MODEL4_PATCHES,
    }
    key = (model_type, data_type)
    if key not in mapping:
        raise ValueError(f"No checkpoint configured for {key}")
    return mapping[key]


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_TYPE}, Data: {DATA_TYPE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    view_names = ['L_CC', 'R_CC', 'L_MLO', 'R_MLO']

    # --- Dataset ---
    if DATA_TYPE == "whole":
        dataset = MammogramClassificationDataset(
            cancerous_dir=CANCEROUS_TEST_WHOLE,
            cancer_free_dir=CANCER_FREE_TEST_WHOLE,
        )
        cancerous_dir = CANCEROUS_TEST_WHOLE
    elif DATA_TYPE == "whole_v2":
        dataset = MammogramClassificationDataset(
            cancerous_dir=CANCEROUS_TEST_WHOLE_V2,
            cancer_free_dir=CANCER_FREE_TEST_WHOLE_V2,
        )
        cancerous_dir = CANCEROUS_TEST_WHOLE_V2
    else:  # patches
        dataset = PatchClassificationDataset(
            cancer_dir=CANCER_TEST_PATCHES,
            cancer_free_dir=CANCER_FREE_TEST_PATCHES,
        )
        cancerous_dir = CANCER_TEST_PATCHES

    print(f"Test samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=False)

    # --- Model ---
    checkpoint = get_checkpoint(MODEL_TYPE, DATA_TYPE)
    full_model = build_model(MODEL_TYPE)
    full_model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)
    )
    full_model.to(device)
    full_model.eval()
    print(f"Loaded weights from {checkpoint}")

    # --- GradCAM setup ---
    single_enc   = SingleViewEncoder(full_model).to(device)
    target_layer = get_target_layer(full_model, MODEL_TYPE)
    gradcam      = GradCAM3D(single_enc, target_layer)
    use_per_slice = MODEL_TYPE in PER_SLICE_MODELS
    print(f"GradCAM target: {MODEL_TYPE} → {type(target_layer).__name__} "
          f"| per-slice={use_per_slice}")

    true_positives = 0

    for views, pad_mask, labels, patients in loader:
        if labels[0].item() != 1:
            continue

        views_dev = views.to(device)
        with torch.no_grad():
            logit = full_model(views_dev)
        prob = torch.sigmoid(logit).item()

        if prob < 0.5:
            print(f"  Skipping {patients[0]} — false negative (prob={prob:.3f})")
            continue

        patient_id = patients[0]
        print(f"\nGenerating GradCAM for patient {patient_id} (prob={prob:.3f})")
        true_positives += 1

        masks_dict = {}
        if DATA_TYPE in ("whole", "whole_v2"):
            masks_dict = load_masks_for_patient(patient_id, cancerous_dir, view_names)
        elif DATA_TYPE == "patches":
            masks_dict = load_masks_for_patient_patches(patient_id, cancerous_dir, view_names)

        cam_dict = {}
        real_t_per_view = pad_mask[0].sum(dim=1).tolist()

        for vi, view_name in enumerate(view_names):
            single_view = views[:, vi, :, :, :].to(device)
            n_real_v = int(real_t_per_view[vi])
            try:
                with torch.enable_grad():
                    if use_per_slice:
                        cam = gradcam.generate_per_slice(single_view, n_real_v)
                    else:
                        cam = gradcam.generate(single_view)
                cam_dict[view_name] = cam
                print(f"  {view_name}: cam={cam.shape} "
                      f"min={cam.min():.3f} max={cam.max():.3f}")
            except Exception as e:
                print(f"  [WARNING] GradCAM failed for {view_name}: {e}")
                cam_dict[view_name] = None

        save_gradcam_per_view(
            views[0], pad_mask[0], cam_dict,
            patient_id, masks_dict, OUTPUT_DIR,
            prob, view_names
        )

    gradcam.remove_hooks()
    print(f"\nDone. GradCAM generated for {true_positives} true positive patients.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()