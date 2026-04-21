"""
resnet50_3d.py

Model 4: ResNet50 3D encoder for mammogram sequence classification.
Based on: https://github.com/ZFTurbo/classification_models_3D

Only the encoder (feature extraction) part is used — the original
classification head is replaced with our shared-weight multi-view
classifier head, consistent with Models 1, 2, and 3.

Changes from original:
- DownTransition uses stride=(1,2,2) in spatial blocks to preserve
  temporal dimension T while downsampling H and W only
- Global average pooling + projection replaces original dense head
- Shared-weight encoding across 4 mammogram views
"""

import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, spatial_stride=1,
                 downsample=None):
        super().__init__()
        # stride=(1, spatial_stride, spatial_stride) preserves T
        s = (1, spatial_stride, spatial_stride) if spatial_stride > 1 else stride
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1   = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=s, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3   = nn.BatchNorm3d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50_3D_Encoder(nn.Module):
    """
    ResNet-50 3D encoder adapted for mammogram sequences.

    Key modification: all spatial downsampling uses stride=(1,2,2)
    to preserve the temporal dimension T (exam sequence axis).

    Input:  [B, 1, T, H, W]
    Output: [B, embed_dim]
    """

    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        # stem — only spatial downsampling
        self.conv1 = nn.Conv3d(in_channels, 64,
                               kernel_size=(3, 7, 7),
                               stride=(1, 2, 2),
                               padding=(1, 3, 3), bias=False)
        self.bn1   = nn.BatchNorm3d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        self.layer1 = self._make_layer(64,  64,  3, spatial_stride=1)
        self.layer2 = self._make_layer(256, 128, 4, spatial_stride=2)
        self.layer3 = self._make_layer(512, 256, 6, spatial_stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, spatial_stride=2)

        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def _make_layer(self, in_planes, planes, blocks, spatial_stride=1):
        downsample = None
        s = (1, spatial_stride, spatial_stride) if spatial_stride > 1 else 1
        out_planes = planes * Bottleneck3D.expansion
        if spatial_stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1,
                          stride=s, bias=False),
                nn.BatchNorm3d(out_planes),
            )
        layers = [Bottleneck3D(in_planes, planes,
                               spatial_stride=spatial_stride,
                               downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(Bottleneck3D(out_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        return self.proj(x)


class MammogramResNet50Classifier(nn.Module):
    """
    ResNet-50 3D mammogram classifier.

    Shared-weight encoder processes each of the 4 views independently.
    View embeddings are averaged then passed to classification head.

    Input:  views [B, 4, 1, T, H, W]
    Output: logits [B]
    """

    def __init__(self, num_views=4, embed_dim=256):
        super().__init__()
        self.num_views = num_views
        self.encoder   = ResNet50_3D_Encoder(in_channels=1,
                                              embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, views):
        B = views.shape[0]
        feats = []
        for v in range(self.num_views):
            x = views[:, v, :, :, :, :]   # [B, 1, T, H, W]
            feats.append(self.encoder(x))
        feat = torch.stack(feats, dim=1).mean(dim=1)   # [B, embed_dim]
        return self.classifier(feat).squeeze(1)         # [B]


if __name__ == '__main__':
    model = MammogramResNet50Classifier(num_views=4, embed_dim=256)
    x = torch.zeros(2, 4, 1, 6, 64, 64)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")