"""Model factory — keeps scripts agnostic of architectural details."""
from __future__ import annotations

from typing import Tuple

from torch import nn

from src.models.custom_cnn import CustomCNN
from src.models.transfer_resnet import build_resnet50


def build_model(name: str, in_channels: int = 3, num_classes: int = 2) -> Tuple[nn.Module, nn.Module]:
    """Return (model, gradcam_target_layer)."""
    name = name.lower()
    if name == "custom_cnn":
        m = CustomCNN(in_channels=in_channels, num_classes=num_classes)
        return m, m.gradcam_layer
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, in_channels=in_channels, pretrained=True)
    raise ValueError(f"Unknown model '{name}'. Expected 'custom_cnn' or 'resnet50'.")


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    if name == "custom_cnn":
        return model.gradcam_layer  # type: ignore[attr-defined]
    if name == "resnet50":
        return model.layer4[-1]  # type: ignore[attr-defined]
    raise ValueError(f"Unknown model '{name}'")
