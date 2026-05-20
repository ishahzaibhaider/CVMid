"""Shared helpers for the classical-CV demos.

Keeping figure styling, output paths, sample-image loading and the random seed
in one place means every unit module stays short and every figure in the report
looks consistent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless: we only ever save figures, never show them
import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "classical"
SAMPLES_DIR = PROJECT_ROOT / "app" / "samples"

# One global seed so every demo is bit-for-bit reproducible across runs.
SEED = 42


def rng() -> np.random.Generator:
    """A fresh, identically-seeded NumPy random generator."""
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------- #
# Figure helpers
# --------------------------------------------------------------------------- #
def set_style() -> None:
    """Apply a clean, report-friendly matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "font.size": 10,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
            "image.cmap": "gray",
        }
    )


def savefig(fig: plt.Figure, name: str) -> Path:
    """Save ``fig`` into the classical figures directory and report the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"    saved  {out.relative_to(PROJECT_ROOT)}")
    return out


# --------------------------------------------------------------------------- #
# Sample images
# --------------------------------------------------------------------------- #
def scene_rgb(max_side: int = 384) -> np.ndarray:
    """A general-purpose colour photo (the classic 'astronaut'), float [0, 1]."""
    img = img_as_float(data.astronaut())
    return _cap(img, max_side)


def scene_gray(name: str = "camera", max_side: int = 384) -> np.ndarray:
    """A grayscale photo. ``name`` is any single-channel ``skimage.data`` image."""
    img = img_as_float(getattr(data, name)())
    if img.ndim == 3:
        img = rgb2gray(img)
    return _cap(img, max_side)


def coins() -> np.ndarray:
    """Grayscale 'coins' image — circular objects, ideal for Hough/blob demos."""
    return img_as_float(data.coins())


def coffee() -> np.ndarray:
    """Colour 'coffee' photo — busy texture, good for segmentation/features."""
    return img_as_float(data.coffee())


def _cap(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is at most ``max_side`` (keeps demos fast)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img.astype(np.float64)
    scale = max_side / longest
    out_shape = (int(round(h * scale)), int(round(w * scale)))
    if img.ndim == 3:
        out_shape = (*out_shape, img.shape[2])
    return resize(img, out_shape, anti_aliasing=True)


def verivision_samples() -> Dict[str, list]:
    """Load VeriVision's bundled REAL/FAKE sample images (CIFAKE 32x32 JPGs).

    Returns a dict ``{"REAL": [arrays], "FAKE": [arrays]}``. These tie the
    classical demos back to the project's own data so the report stays coherent.
    Falls back to an empty list if the samples folder is missing.
    """
    from PIL import Image

    out: Dict[str, list] = {"REAL": [], "FAKE": []}
    for cls in out:
        for p in sorted(SAMPLES_DIR.glob(f"{cls.lower()}_*.jpg")):
            out[cls].append(np.asarray(Image.open(p).convert("RGB")))
    return out


def add_gaussian_noise(img: np.ndarray, sigma: float = 0.08) -> np.ndarray:
    """Add zero-mean Gaussian noise and clip back to [0, 1]."""
    noisy = img + rng().normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0.0, 1.0)
