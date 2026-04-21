"""
vnet3d.py

VNet encoder adapted for mammogram sequence classification.
Based on: https://github.com/black0017/MedicalZooPytorch

Changes from original:
- DownTransition uses kernel_size=(1,2,2) stride=(1,2,2) to preserve
  temporal dimension (T) while downsampling spatial (H, W) only
- Decoder (UpTransition, OutputTransition) removed
- Global average pooling + projection added after encoder
- Shared-weight encoding across 4 mammogram views
- Classification head for binary output
"""

import torch
import torch.nn as nn


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super().__init__()
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(nchan)
        self.relu1 = ELUCons(elu, nchan)

    def forward(self, x):
        return self.relu1(self.bn1(self.conv1(x)))


def _make_nConv(nchan, depth, elu):
    return nn.Sequential(*[LUConv(nchan, elu) for _ in range(depth)])


class InputTransition(nn.Module):
    def __init__(self, in_channels, elu):
        super().__init__()
        self.num_features = 16
        self.in_channels  = in_channels
        self.conv1 = nn.Conv3d(in_channels, self.num_features,
                               kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(self.num_features)
        self.relu1 = ELUCons(elu, self.num_features)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        repeat_rate = self.num_features // self.in_channels
        x16 = x.repeat(1, repeat_rate, 1, 1, 1)
        return self.relu1(torch.add(out, x16))


class DownTransition(nn.Module):
    """
    Downsampling block with (1,2,2) stride to preserve temporal dimension.
    """
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super().__init__()
        outChans = 2 * inChans
        # Only downsample H and W, keep T intact
        self.down_conv = nn.Conv3d(inChans, outChans,
                                   kernel_size=(1, 2, 2),
                                   stride=(1, 2, 2))
        self.bn1   = nn.BatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        self.do1   = nn.Dropout3d() if dropout else passthrough
        self.ops   = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out  = self.do1(down)
        out  = self.ops(out)
        return self.relu2(torch.add(out, down))


class VNetEncoder(nn.Module):
    """
    VNet encoder — produces a 256-dim feature vector per view.
    Input:  [B, 1, T, H, W]
    Output: [B, 256]
    """
    def __init__(self, in_channels=1, elu=True, embed_dim=256):
        super().__init__()
        self.in_tr      = InputTransition(in_channels, elu)   # → 16ch
        self.down_tr32  = DownTransition(16,  1, elu)          # → 32ch
        self.down_tr64  = DownTransition(32,  2, elu)          # → 64ch
        self.down_tr128 = DownTransition(64,  3, elu, dropout=True)  # → 128ch
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)  # → 256ch

        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.in_tr(x)
        x = self.down_tr32(x)
        x = self.down_tr64(x)
        x = self.down_tr128(x)
        x = self.down_tr256(x)
        x = self.gap(x)
        x = self.proj(x)
        return x


class MammogramVNetClassifier(nn.Module):
    """
    VNet-based mammogram classifier.

    Shared-weight encoder processes each of the 4 views independently.
    View embeddings are averaged then passed to classification head.

    Input:  views [B, 4, 1, T, H, W]
    Output: logits [B]
    """
    def __init__(self, num_views=4, embed_dim=256, elu=True):
        super().__init__()
        self.num_views = num_views
        self.encoder   = VNetEncoder(in_channels=1, elu=elu,
                                     embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, views):
        # views: [B, 4, 1, T, H, W]
        B = views.shape[0]
        feats = []
        for v in range(self.num_views):
            x = views[:, v, :, :, :, :]  # [B, 1, T, H, W]
            feats.append(self.encoder(x))
        # Mean pooling across views
        feat = torch.stack(feats, dim=1).mean(dim=1)  # [B, embed_dim]
        return self.classifier(feat).squeeze(1)        # [B]


if __name__ == '__main__':
    model = MammogramVNetClassifier(num_views=4, embed_dim=256)
    x = torch.zeros(2, 4, 1, 6, 64, 64)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")