"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper: https://arxiv.org/abs/1606.06650

Fixes over original:
- UpConv3DBlock now uses two separate BatchNorm3d instances (bn1, bn2)
  instead of one shared instance applied twice to different tensors.
- Default level_channels reduced to [16, 32, 64] / bottleneck 128
  to avoid overfitting on small medical datasets (~200 positive patients).
  Pass level_channels=[64,128,256], bottleneck_channel=512 to restore
  the original paper capacity if you have more data.
"""

import torch
from torch import nn


class Conv3DBlock(nn.Module):
    """
    Encoder block: two 3x3x3 convolutions + optional MaxPool.
    Pooling uses kernel (1,2,2) to preserve the temporal (T) dimension
    while downsampling spatial (H, W) dimensions only.

    Returns (pooled_output, pre_pool_residual).
    """

    def __init__(self, in_channels, out_channels, bottleneck=False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(out_channels // 2)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.bottleneck = bottleneck
        if not bottleneck:
            # Only downsample H and W, keep T intact
            self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        residual = x
        if not self.bottleneck:
            x = self.pool(x)
        return x, residual


class UpConv3DBlock(nn.Module):
    """
    Decoder block: transposed conv upsample + skip connection + two 3x3x3 convs.

    Fix: uses two independent BatchNorm3d instances (bn1, bn2) instead of
    the original single shared instance, which incorrectly merged running
    statistics and affine parameters from two different feature maps.
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None):
        super().__init__()
        assert (not last_layer and num_classes is None) or \
               (last_layer and num_classes is not None), \
               'last_layer=True requires num_classes, last_layer=False requires num_classes=None'

        mid = in_channels // 2

        # Upsample only spatial dims
        self.upconv = nn.ConvTranspose3d(in_channels, in_channels,
                                         kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1  = nn.Conv3d(in_channels + res_channels, mid, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm3d(mid)   # independent BN for conv1 output
        self.conv2  = nn.Conv3d(mid, mid, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm3d(mid)   # independent BN for conv2 output
        self.relu   = nn.ReLU(inplace=True)

        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(mid, num_classes, kernel_size=1)

    def forward(self, x, residual=None):
        x = self.upconv(x)
        if residual is not None:
            x = torch.cat([x, residual], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.last_layer:
            x = self.conv3(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for spatiotemporal segmentation.

    Input:  [B, in_channels, T, H, W]
            T = temporal depth (exam sequence, padded to MAX_T)
            H, W = spatial dimensions (e.g. 256x256)

    Output: [B, num_classes, T, H, W]  (same shape as input)

    Default capacity is intentionally small (level_channels=[16,32,64])
    to avoid overfitting on datasets with ~150-200 positive patients.
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        level_channels=(16, 32, 64),
        bottleneck_channel=128,
    ):
        super().__init__()
        l1, l2, l3 = level_channels

        # Encoder
        self.enc1      = Conv3DBlock(in_channels, l1)
        self.enc2      = Conv3DBlock(l1, l2)
        self.enc3      = Conv3DBlock(l2, l3)
        self.bottleneck = Conv3DBlock(l3, bottleneck_channel, bottleneck=True)

        # Decoder
        self.dec3 = UpConv3DBlock(bottleneck_channel, res_channels=l3)
        self.dec2 = UpConv3DBlock(l3,                 res_channels=l2)
        self.dec1 = UpConv3DBlock(l2,                 res_channels=l1,
                                  last_layer=True, num_classes=num_classes)

    def forward(self, x):
        # Encoder
        x, r1 = self.enc1(x)
        x, r2 = self.enc2(x)
        x, r3 = self.enc3(x)
        x, _  = self.bottleneck(x)

        # Decoder
        x = self.dec3(x, r3)
        x = self.dec2(x, r2)
        x = self.dec1(x, r1)
        return x


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    model = UNet3D(in_channels=1, num_classes=1)
    x = torch.zeros(2, 1, 6, 256, 256)   # batch=2, T=6, 256x256
    out = model(x)
    print("Input: ", x.shape)
    print("Output:", out.shape)
    assert out.shape == x.shape, "Output shape mismatch!"
    print("OK")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")