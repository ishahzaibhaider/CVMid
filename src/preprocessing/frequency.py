"""Frequency-domain preprocessing.

Diffusion models leave characteristic fingerprints in the Fourier spectrum
(Bird & Lotfi 2024, UGAD 2024). We expose two utilities:

* `compute_fft_magnitude(x)`   — numpy-friendly helper for notebooks and the demo app.
* `FFTChannel`                 — a torchvision-style transform that appends the
                                  log-magnitude spectrum of the luminance channel
                                  as a 4th tensor channel, so the model learns
                                  jointly from pixel and spectral features.

Why log-magnitude? The raw FFT magnitude has a very heavy-tailed distribution.
`log1p` squashes the DC component and makes high-frequency content visible,
which is where synthetic-image artifacts tend to concentrate.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch


def compute_fft_magnitude(img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Return a 2D log-magnitude spectrum normalized to [0, 1].

    Accepts HWC uint8 numpy arrays or CHW float tensors in [0, 1].
    """
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = img
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    if arr.ndim == 3:
        # Rec. 601 luminance — matches what the human visual system is sensitive to
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr
    fft = np.fft.fft2(gray)
    fft = np.fft.fftshift(fft)
    mag = np.log1p(np.abs(fft))
    m_min, m_max = float(mag.min()), float(mag.max())
    if m_max - m_min < 1e-8:
        return np.zeros_like(mag, dtype=np.float32)
    return ((mag - m_min) / (m_max - m_min)).astype(np.float32)


class FFTChannel:
    """Transform that appends an FFT magnitude channel to an already-tensored image.

    Intended to sit AFTER ToTensor() in the transform pipeline. Produces a
    4-channel tensor (R, G, B, FFT) where FFT is in [0, 1].
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Expected a 3xHxW tensor, got shape {tuple(tensor.shape)}")
        # Luminance channel in torch land — stays differentiable if ever needed
        gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
        fft = torch.fft.fft2(gray)
        fft = torch.fft.fftshift(fft)
        mag = torch.log1p(fft.abs())
        m_min, m_max = mag.min(), mag.max()
        # Guard against constant images (solid color patches in CIFAKE do exist)
        if (m_max - m_min).item() < 1e-8:
            fft_channel = torch.zeros_like(mag)
        else:
            fft_channel = (mag - m_min) / (m_max - m_min)
        return torch.cat([tensor, fft_channel.unsqueeze(0)], dim=0)
