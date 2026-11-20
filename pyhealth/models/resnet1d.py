"""
ResNet1D backbone + PyHealth BaseModel stencil.

Backbone adapted from Hong et al. (hsd1503/resnet1d):
  Input:  (batch, in_channels, seq_len)
  Output: (batch, n_classes)  # or hidden_size when used as an encoder
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pyhealth.datasets import SampleDataset
    from pyhealth.models import BaseModel

    _HAS_PYHEALTH = True
except ImportError:  # backbone still usable without PyHealth installed
    SampleDataset = object  # type: ignore
    BaseModel = nn.Module  # type: ignore
    _HAS_PYHEALTH = False


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    """ResNet basic residual block for 1D signals."""

    def __init__(
        self,
        in_channels,
        out_channels,
        downsample
    ):
    #to do figure out dropout
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        out = self.bn2(x)
        if self.downsample or self.in_channels != self.out_channels:
            out = self.shortcut(out)
        else:
            out = out + x
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """1D ResNet backbone.

    Args:
        in_channels: number of input channels (e.g. ECG leads).
        base_filters: filters in the first conv; doubles every ``increasefilter_gap`` blocks.
        kernel_size: conv kernel width.
        stride: stride used when a block downsamples.
        groups: 1 = ResNet; >1 ≈ ResNeXt-style grouped conv.
        n_block: number of residual blocks.
        n_classes: output dim (num classes, or hidden size if used as encoder).
    """

    def __init__(
        self,
        in_channels,
        n_classes,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = BasicBlock(64, 64, False)
        self.conv2_2 = BasicBlock(64, 64, False)
        
        self.conv3_1 = BasicBlock(64, 128, True)
        self.conv3_2 = BasicBlock(128, 128, False)

        self.conv4_1 = BasicBlock(128, 256, True)
        self.conv4_2 = BasicBlock(256, 256, False)

        self.conv5_1 = BasicBlock(256, 512, True)
        self.conv5_2 = BasicBlock(512, 512, False)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, n_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# PyHealth model stencil
# ---------------------------------------------------------------------------

class ResNet1DModel(BaseModel):
    """PyHealth stencil wrapping :class:`ResNet1D`.

    Expects a single timeseries / tensor feature shaped
    ``(batch, channels, seq_len)`` (or ``(batch, seq_len, channels)`` if
    ``channels_last=True``).

    Fill in / adjust:
      - ``feature_key``: which dataset field holds the 1D signal
      - channel layout (channels_first vs channels_last)
      - ResNet hyperparameters below
    """

    def __init__(
        self,
        classes:int,
        in_channels: int = 12
    ):
        super().__init__()
        self.resnet1d = ResNet1D(
            in_channels,
            classes
        )
        
    def forward(self, x):
        return self.resnet1d(x)

        


if __name__ == "__main__":
    # Smoke test the backbone only (no PyHealth dataset required)
    net = ResNet1D(
        in_channels=12,
        base_filters=32,
        kernel_size=16,
        stride=2,
        groups=1,
        n_block=4,
        n_classes=5,
    )
    y = net(torch.randn(2, 12, 1000))
    print("ResNet1D out:", y.shape)  # (2, 5)
