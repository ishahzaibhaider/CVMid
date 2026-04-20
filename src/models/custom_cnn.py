"""Baseline CNN trained from scratch on 32x32 CIFAKE.

Roughly modelled on the architecture in Bird & Lotfi (2024) — two conv blocks,
global average pooling, small MLP head. Parameter count stays under 1M so it
trains comfortably on a laptop CPU/MPS.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return self.pool(x)


class CustomCNN(nn.Module):
    """Compact CNN for 32x32 binary classification."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # 32 -> 16
            ConvBlock(32, 64),            # 16 -> 8
            ConvBlock(64, 128),           # 8 -> 4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

    @property
    def gradcam_layer(self) -> nn.Module:
        # Last conv block's final conv — matches the Grad-CAM sweet spot
        return self.features[-1].conv2
