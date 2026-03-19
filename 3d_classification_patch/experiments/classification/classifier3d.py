"""
classifier3d.py

Lightweight 3D CNN for binary classification of mammogram sequences.

Architecture:
- Shared-weight 3D CNN encoder processes each view sequence independently
- Input per view: [B, 1, T, H, W] where T = temporal depth (exam sequence)
- The 4 view embeddings are mean-pooled into one patient embedding
- Classification head outputs one probability per patient

This is intentionally much smaller than the UNet3D — appropriate for
~150 positive training patients.
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Conv3D → BN → ReLU → Conv3D → BN → ReLU → MaxPool"""

    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) if pool else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.pool is not None:
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
        self.block1 = ConvBlock3D(in_channels, 32,  pool=True)   # H/2,  W/2
        self.block2 = ConvBlock3D(32,          64,  pool=True)   # H/4,  W/4
        self.block3 = ConvBlock3D(64,          128, pool=True)   # H/8,  W/8
        self.block4 = ConvBlock3D(128,         256, pool=True)   # H/16, W/16

        # Global average pooling over T, H, W → [B, 256]
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
        x = self.gap(x)           # [B, 256, 1, 1, 1]
        x = x.flatten(1)          # [B, 256]
        x = self.proj(x)          # [B, embed_dim]
        return x


class MammogramClassifier(nn.Module):
    """
    Per-patient binary classifier using shared-weight encoder across views.

    Input:  list of view tensors, each [B, 1, T, H, W]
            OR a single tensor [B, V, 1, T, H, W] where V = num views
    Output: [B] logits (apply sigmoid for probability)

    Usage:
        model = MammogramClassifier(num_views=4)
        logits = model(views)   # views: [B, 4, 1, T, H, W]
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
        """
        views: [B, V, 1, T, H, W]
        """
        B, V, C, T, H, W = views.shape
        assert V == self.num_views, f"Expected {self.num_views} views, got {V}"

        # Encode each view independently with shared weights
        # Reshape to [B*V, C, T, H, W], encode, reshape back to [B, V, embed_dim]
        x = views.view(B * V, C, T, H, W)
        x = self.encoder(x)                  # [B*V, embed_dim]
        x = x.view(B, V, -1)                 # [B, V, embed_dim]

        # Mean-pool across views → [B, embed_dim]
        x = x.mean(dim=1)

        # Classify
        logits = self.classifier(x)          # [B, 1]
        return logits.squeeze(1)             # [B]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model = MammogramClassifier(num_views=4, embed_dim=256)
    x = torch.zeros(2, 4, 1, 6, 512, 512)   # batch=2, 4 views, T=6, 512x512
    out = model(x)
    print("Input: ", x.shape)
    print("Output:", out.shape)              # should be [2]
    assert out.shape == (2,)
    print("OK")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")