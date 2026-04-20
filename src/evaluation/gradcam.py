"""Grad-CAM wrapper.

Uses jacobgil's `pytorch-grad-cam` when available; falls back to a small
hand-rolled Grad-CAM implementation so the project stays runnable even if
the optional dependency isn't installed.

Why it matters for this project: the Bird & Lotfi CIFAKE paper found that
detectors focus on *background texture* rather than foreground objects —
Grad-CAM is what surfaces that finding visually. It's the difference between
"the model got 95% accuracy" (nice) and "here's *why* — look at how it fixates
on these sky gradients" (a presentation).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

try:  # pragma: no cover — optional dependency
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _HAVE_JACOB = True
except Exception:  # pragma: no cover
    _HAVE_JACOB = False


def _hand_rolled_gradcam(
    model: nn.Module, target_layer: nn.Module, input_tensor: torch.Tensor, target_class: int
) -> np.ndarray:
    """Minimal Grad-CAM fallback."""
    model.eval()
    activations: list = []
    gradients: list = []

    def fwd_hook(_module, _inp, out):
        activations.append(out.detach())

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        logits = model(input_tensor)
        score = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        act = activations[0][0]          # (C, H, W)
        grad = gradients[0][0]           # (C, H, W)
        weights = grad.mean(dim=(1, 2))  # GAP over spatial dims → per-channel weights
        cam = torch.einsum("c,chw->hw", weights, act)
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def gradcam_overlay(
    model: nn.Module,
    target_layer: nn.Module,
    input_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    target_class: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap and overlay it on the original RGB image.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Shape (1, C, H, W), already on the right device and normalized.
    original_rgb : np.ndarray
        Shape (H, W, 3), uint8 or float in [0, 1]. The *display* image — we
        keep it separate from `input_tensor` because the latter may be
        normalized/upsampled/4-channel.
    """
    model.eval()
    if target_class is None:
        with torch.no_grad():
            target_class = int(model(input_tensor).argmax(dim=1).item())

    if _HAVE_JACOB:
        cam_extractor = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam_extractor(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(target_class)],
        )[0]
    else:
        grayscale_cam = _hand_rolled_gradcam(model, target_layer, input_tensor, target_class)

    rgb = original_rgb.astype(np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    # Upsample CAM to match the display image's spatial size
    cam_resized = _resize(grayscale_cam, rgb.shape[:2])
    heatmap = plt.get_cmap("jet")(cam_resized)[..., :3]
    overlay = 0.4 * heatmap + 0.6 * rgb
    overlay = np.clip(overlay, 0, 1)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        axes[0].imshow(rgb)
        axes[0].set_title("Input")
        axes[1].imshow(cam_resized, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return overlay


def _resize(arr: np.ndarray, shape) -> np.ndarray:
    """Bilinear resize via numpy/PIL to avoid a hard cv2 dependency."""
    from PIL import Image

    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0
