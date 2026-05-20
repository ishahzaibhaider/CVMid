"""Unit 2a — Sampling, aliasing and multiscale image representations.

Covers: image sampling & the aliasing it causes, anti-alias filtering,
up/down-sampling with different interpolation orders, and Gaussian/Laplacian
image pyramids (the multiscale representations features are detected on).

Run:  python -m src.classical.unit2_sampling
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import pyramid_gaussian, pyramid_laplacian, resize

from src.classical.common import savefig, scene_gray, set_style


# --------------------------------------------------------------------------- #
# 1. Sampling and aliasing
# --------------------------------------------------------------------------- #
def _zone_plate(n: int = 512) -> np.ndarray:
    """A 'zone plate': spatial frequency rises with radius.

    It is the standard stress-test for sampling because its outer rings exceed
    any finite sampling rate — exactly where aliasing becomes visible.
    """
    x = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(x, x)
    r2 = xx ** 2 + yy ** 2
    return 0.5 + 0.5 * np.cos(n * 1.15 * r2)


def aliasing_figure() -> None:
    """Down-sample a zone plate with and without an anti-alias pre-filter."""
    img = _zone_plate(512)
    factor = 6
    small = (img.shape[0] // factor, img.shape[1] // factor)

    naive = img[::factor, ::factor]                       # just drop samples
    filtered = resize(img, small, anti_aliasing=True)     # low-pass, then sample

    # Blow both back up (nearest) so the moire pattern is visible on the page.
    show = lambda a: resize(a, img.shape, order=0, anti_aliasing=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img); axes[0].set_title("Original signal\n(frequency rises outward)")
    axes[1].imshow(show(naive))
    axes[1].set_title(f"Sub-sampled 1/{factor}  — NO filter\n(false low-freq moire rings)")
    axes[2].imshow(show(filtered))
    axes[2].set_title(f"Sub-sampled 1/{factor}  — anti-aliased\n(low-pass first: artefacts gone)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Sampling & aliasing — why pre-filtering before down-sampling matters",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit2_aliasing.png")


# --------------------------------------------------------------------------- #
# 2. Up-sampling — interpolation order matters
# --------------------------------------------------------------------------- #
def resampling_figure() -> None:
    """Down-sample an image hard, then up-sample with nearest vs bicubic."""
    img = scene_gray("camera")
    tiny = resize(img, (32, 32), anti_aliasing=True)         # heavy down-sample
    big = (256, 256)
    nearest = resize(tiny, big, order=0, anti_aliasing=False)
    bilinear = resize(tiny, big, order=1, anti_aliasing=False)
    bicubic = resize(tiny, big, order=3, anti_aliasing=False)

    panels = [
        (tiny, "Down-sampled to 32x32\n(information bottleneck)"),
        (nearest, "Up-sample: nearest (order 0)\nblocky"),
        (bilinear, "Up-sample: bilinear (order 1)\nsmoother"),
        (bicubic, "Up-sample: bicubic (order 3)\nsmoothest"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im); ax.set_title(title); ax.axis("off")
    fig.suptitle("Up-sampling cannot create detail — interpolation only changes smoothness",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit2_resampling.png")


# --------------------------------------------------------------------------- #
# 3. Image pyramids — multiscale representation
# --------------------------------------------------------------------------- #
def pyramids_figure() -> None:
    """Build Gaussian and Laplacian pyramids of an image."""
    img = scene_gray("camera")
    n_levels = 4
    gauss = list(pyramid_gaussian(img, max_layer=n_levels, downscale=2,
                                  channel_axis=None))
    lapl = list(pyramid_laplacian(img, max_layer=n_levels, downscale=2,
                                  channel_axis=None))

    fig, axes = plt.subplots(2, n_levels + 1, figsize=(13, 5.2))
    for j, level in enumerate(gauss):
        axes[0, j].imshow(level)
        axes[0, j].set_title(f"Gaussian L{j}\n{level.shape[0]}x{level.shape[1]}")
        axes[0, j].axis("off")
    for j, level in enumerate(lapl):
        # Laplacian levels are band-pass (signed); centre at 0.5 for display.
        axes[1, j].imshow(level + 0.5, vmin=0, vmax=1)
        axes[1, j].set_title(f"Laplacian L{j}\n(band-pass detail)")
        axes[1, j].axis("off")
    fig.suptitle("Multiscale image representations — Gaussian (coarse) & Laplacian (detail) pyramids",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit2_pyramids.png")


def demo() -> None:
    print("[Unit 2a] Sampling, aliasing & multiscale representations")
    set_style()
    aliasing_figure()
    resampling_figure()
    pyramids_figure()


if __name__ == "__main__":
    demo()
