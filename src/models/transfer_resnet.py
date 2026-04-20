"""ResNet-50 transfer-learning head for CIFAKE.

CIFAKE is 32x32, but ImageNet-pretrained stems expect 224x224; feeding the raw
32x32 directly discards most of the pretrained features because the stem's
stride-2 convolutions collapse the signal immediately. Standard practice is to
upsample to 224 and let the pretrained feature extractor do its job. Transforms
in src.preprocessing.transforms already handle the resize.

When `in_channels != 3` (i.e. we're using the FFT 4th channel), we surgically
replace the first conv to accept the extra channel and initialize the new weight
slice with the mean of the RGB filters — a standard trick that preserves most
of the pretrained signal.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models


def build_resnet50(
    num_classes: int = 2, in_channels: int = 3, pretrained: bool = True
) -> Tuple[nn.Module, nn.Module]:
    """Return (model, gradcam_target_layer)."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            # Copy RGB weights as-is, init extra channels from channel-mean
            new_conv.weight[:, :3] = old_conv.weight
            if in_channels > 3:
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:] = mean_w.repeat(1, in_channels - 3, 1, 1)
        model.conv1 = new_conv

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    # layer4 final block is the Grad-CAM sweet spot for ResNet-50
    target_layer = model.layer4[-1]
    return model, target_layer
