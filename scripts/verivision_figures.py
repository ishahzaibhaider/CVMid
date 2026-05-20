"""Regenerate the VeriVision data figures from the bundled sample images.

Unlike ``scripts/demo.py`` (which needs the full CIFAKE dataset and a trained
PyTorch checkpoint), this script is dependency-light: it works from the ten
sample JPGs in ``app/samples/`` so the report's data figures can always be
rebuilt. It produces:

    reports/figures/verivision_samples.png   — REAL vs FAKE sample grid
    reports/figures/verivision_fft.png       — mean FFT spectra + fingerprint
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = PROJECT_ROOT / "app" / "samples"
FIGURES = PROJECT_ROOT / "reports" / "figures"


def _load(cls: str) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB"))
            for p in sorted(SAMPLES.glob(f"{cls}_*.jpg"))]


def _fft_magnitude(rgb: np.ndarray) -> np.ndarray:
    """Log-magnitude of the centred 2D FFT of the luminance channel, [0, 1]."""
    arr = rgb.astype(np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    lo, hi = float(mag.min()), float(mag.max())
    return (mag - lo) / (hi - lo + 1e-8)


def samples_figure(real: list, fake: list) -> None:
    n = min(len(real), len(fake))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3.4))
    for j in range(n):
        axes[0, j].imshow(real[j]); axes[0, j].axis("off")
        axes[1, j].imshow(fake[j]); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("REAL")
    axes[0, 0].set_title("REAL (camera)", loc="left", fontsize=10)
    axes[1, 0].set_title("FAKE (Stable Diffusion)", loc="left", fontsize=10)
    fig.suptitle("CIFAKE samples — REAL vs AI-generated (32x32 RGB)",
                 fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "verivision_samples.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved  {out.relative_to(PROJECT_ROOT)}")


def fft_figure(real: list, fake: list) -> None:
    spec_real = np.mean([_fft_magnitude(im) for im in real], axis=0)
    spec_fake = np.mean([_fft_magnitude(im) for im in fake], axis=0)
    diff = spec_fake - spec_real
    lim = float(np.abs(diff).max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(spec_real, cmap="magma"); axes[0].set_title("Mean FFT — REAL")
    axes[1].imshow(spec_fake, cmap="magma"); axes[1].set_title("Mean FFT — FAKE")
    im = axes[2].imshow(diff, cmap="seismic", vmin=-lim, vmax=lim)
    axes[2].set_title("FAKE - REAL\n(spectral fingerprint)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Frequency-domain analysis — diffusion models leave spectral "
                 "fingerprints", fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "verivision_fft.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved  {out.relative_to(PROJECT_ROOT)}")


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    real, fake = _load("real"), _load("fake")
    if not real or not fake:
        raise SystemExit(f"No sample images found in {SAMPLES}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("VeriVision data figures:")
    samples_figure(real, fake)
    fft_figure(real, fake)


if __name__ == "__main__":
    main()
