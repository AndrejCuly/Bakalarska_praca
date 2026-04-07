"""
resnet3d.py

3D ResNet-10 architecture from MedicalNet (Tencent).
Adapted for per-patient binary classification with shared-weight
encoding across 4 mammogram views.

Original paper: Med3D: Transfer Learning for 3D Medical Image Analysis
Chen et al., arXiv:1904.00625

Key changes from original MedicalNet:
- Removed segmentation head (conv_seg)
- Added global average pooling + projection to get fixed-size embedding
- Added MammogramResNetClassifier wrapper for 4-view shared-weight encoding
- Added load_pretrained() utility to handle MedicalNet weight format
  (state_dict stored under 'state_dict' key with 'module.' prefix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math


# -------------------------------------------------------------------------
# Building blocks
# -------------------------------------------------------------------------

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1   = conv3x3x3(inplanes, planes, stride)
        self.bn1     = nn.BatchNorm3d(planes)
        self.relu    = nn.ReLU(inplace=True)
        self.conv2   = conv3x3x3(planes, planes)
        self.bn2     = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride  = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


# -------------------------------------------------------------------------
# ResNet encoder (no segmentation head)
# -------------------------------------------------------------------------

class ResNet3D(nn.Module):
    """
    3D ResNet encoder compatible with MedicalNet pretrained weights.

    Input:  [B, 1, T, H, W]
    Output: [B, embed_dim] feature vector via global average pooling

    For ResNet-10: layers = [1, 1, 1, 1], shortcut_type = 'B'
    """

    def __init__(self, block, layers, shortcut_type='B',
                 in_channels=1, embed_dim=256):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7,
                               stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1   = nn.BatchNorm3d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Linear(512 * block.expansion, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)        # [B, 512, 1, 1, 1]
        x = x.flatten(1)       # [B, 512]
        x = self.proj(x)       # [B, embed_dim]
        return x


def resnet10(in_channels=1, embed_dim=256, **kwargs):
    return ResNet3D(BasicBlock, [1, 1, 1, 1],
                    in_channels=in_channels, embed_dim=embed_dim, **kwargs)


# -------------------------------------------------------------------------
# Per-patient classifier with shared-weight encoding across 4 views
# -------------------------------------------------------------------------

class MammogramResNetClassifier(nn.Module):
    """
    Per-patient binary classifier using shared-weight ResNet-10 encoder.

    Input:  [B, V, 1, T, H, W]  (V=4 views)
    Output: [B] logits
    """

    def __init__(self, num_views=4, embed_dim=256, shortcut_type='B'):
        super().__init__()
        self.num_views = num_views
        self.encoder   = resnet10(in_channels=1, embed_dim=embed_dim,
                                  shortcut_type=shortcut_type)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, views):
        B, V, C, T, H, W = views.shape
        assert V == self.num_views

        x = views.view(B * V, C, T, H, W)
        x = self.encoder(x)        # [B*V, embed_dim]
        x = x.view(B, V, -1)       # [B, V, embed_dim]
        x = x.mean(dim=1)          # [B, embed_dim]
        return self.classifier(x).squeeze(1)   # [B]


# -------------------------------------------------------------------------
# Pretrained weight loader
# -------------------------------------------------------------------------

def load_pretrained_resnet(model, pretrained_path, device='cpu'):
    """
    Load MedicalNet pretrained weights into ResNet3D encoder.

    Handles:
    - 'state_dict' wrapper key
    - 'module.' prefix from DataParallel training
    - Mismatched keys (strict=False) — skips final layers not in pretrained
    """
    print(f"Loading pretrained weights from {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device,
                            weights_only=False)

    # MedicalNet stores weights under 'state_dict' key
    if 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    else:
        pretrained_dict = checkpoint

    # Strip 'module.' prefix from DataParallel training
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load into encoder only (skip proj layer — different size)
    encoder_dict   = model.encoder.state_dict()
    pretrained_enc = {k: v for k, v in new_state_dict.items()
                      if k in encoder_dict and
                      encoder_dict[k].shape == v.shape}

    encoder_dict.update(pretrained_enc)
    model.encoder.load_state_dict(encoder_dict, strict=False)

    total   = len(encoder_dict)
    loaded  = len(pretrained_enc)
    skipped = total - loaded
    print(f"  Loaded {loaded}/{total} encoder layers "
          f"({skipped} skipped — size mismatch or new layers)")
    return model


# -------------------------------------------------------------------------
# Sanity check
# -------------------------------------------------------------------------
if __name__ == '__main__':
    model = MammogramResNetClassifier(num_views=4, embed_dim=256)
    x     = torch.zeros(2, 4, 1, 6, 256, 256)
    out   = model(x)
    print("Input: ", x.shape)
    print("Output:", out.shape)
    assert out.shape == (2,)
    print("OK")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")