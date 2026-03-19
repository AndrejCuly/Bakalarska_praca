"""
classifier3d.py

Lightweight 3D CNN for binary classification of mammogram sequences.

Memory optimization: aggressive early spatial downsampling so that
512x512 inputs become manageable quickly, allowing batch_size=8 with
4 views without OOM on a 12GB GPU.

Spatial reduction per block for 512x512 input:
    block1: (1,4,4) pool → T x 128 x 128
    block2: (1,2,2) pool → T x 64  x 64
    block3: (2,2,2) pool → T/2 x 32 x 32
    block4: (2,2,2) pool → T/4 x 16 x 16
    GAP    → [B, 256]
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Conv3D → BN → ReLU → Conv3D → BN → ReLU → MaxPool"""

    def __init__(self, in_channels, out_channels, pool_kernel=(1, 2, 2)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class Encoder3D(nn.Module):
    """
    Lightweight 3D CNN encoder.
    Input:  [B, 1, T, H, W]
    Output: [B, embed_dim] feature vector
    """

    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        self.block1 = ConvBlock3D(in_channels, 32,  pool_kernel=(1, 4, 4))  # 512->128
        self.block2 = ConvBlock3D(32,          64,  pool_kernel=(1, 2, 2))  # 128->64
        self.block3 = ConvBlock3D(64,          128, pool_kernel=(2, 2, 2))  # 64->32, T/2
        self.block4 = ConvBlock3D(128,         256, pool_kernel=(2, 2, 2))  # 32->16, T/4

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


class MammogramClassifier(nn.Module):
    """
    Per-patient binary classifier using shared-weight encoder across views.
    Input:  [B, V, 1, T, H, W]
    Output: [B] logits
    """

    def __init__(self, num_views=4, embed_dim=256):
        super().__init__()
        self.num_views = num_views
        self.encoder   = Encoder3D(in_channels=1, embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, views):
        B, V, C, T, H, W = views.shape
        assert V == self.num_views, f"Expected {self.num_views} views, got {V}"

        x = views.view(B * V, C, T, H, W)
        x = self.encoder(x)
        x = x.view(B, V, -1)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits.squeeze(1)


if __name__ == '__main__':
    model = MammogramClassifier(num_views=4, embed_dim=256)
    x = torch.zeros(2, 4, 1, 6, 512, 512)
    out = model(x)
    print("Input: ", x.shape)
    print("Output:", out.shape)
    assert out.shape == (2,)
    print("OK")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")