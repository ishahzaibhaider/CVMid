"""Unit 3a — Image transformations and image warping.

Covers: the family of 2D geometric transformations (translation, Euclidean,
similarity, affine, projective/homography) and non-rigid image warping. These
are the maps that relate two views of a scene and underpin stereo and stitching.

Run:  python -m src.classical.unit3_transforms
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import (AffineTransform, EuclideanTransform,
                               ProjectiveTransform, SimilarityTransform, swirl,
                               warp)

from src.classical.common import savefig, scene_rgb, set_style


# --------------------------------------------------------------------------- #
# 1. The hierarchy of 2D transformations
# --------------------------------------------------------------------------- #
def transformations_figure() -> None:
    """Apply each transformation class and show how many DoF it adds."""
    img = scene_rgb(max_side=320)
    shape = img.shape
    h, w = shape[:2]

    # Projective transform from a 4-point correspondence (a perspective tilt).
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    dst = np.array([[w * 0.10, h * 0.05], [w * 0.92, h * 0.16],
                    [w * 0.84, h * 0.96], [w * 0.04, h * 0.82]])
    projective = ProjectiveTransform.from_estimate(src, dst)

    transforms = [
        ("Original", None, "6 .. 8 DoF below"),
        ("Translation", EuclideanTransform(translation=(45, 25)), "2 DoF"),
        ("Euclidean\n(rotate + translate)", EuclideanTransform(
            rotation=np.deg2rad(20), translation=(60, -10)), "3 DoF"),
        ("Similarity\n(+ uniform scale)", SimilarityTransform(
            scale=0.75, rotation=np.deg2rad(15), translation=(40, 30)), "4 DoF"),
        ("Affine\n(+ shear, parallel lines kept)", AffineTransform(
            scale=(0.9, 0.9), shear=0.3, translation=(10, 10)), "6 DoF"),
        ("Projective / homography\n(straight lines kept)", projective, "8 DoF"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, (title, tf, dof) in zip(axes.ravel(), transforms):
        out = img if tf is None else warp(img, tf.inverse, output_shape=shape,
                                          cval=1.0)
        ax.imshow(out)
        ax.set_title(f"{title}\n[{dof}]")
        ax.axis("off")
    fig.suptitle("The hierarchy of 2D geometric transformations",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit3_transformations.png")


# --------------------------------------------------------------------------- #
# 2. Non-rigid image warping
# --------------------------------------------------------------------------- #
def warping_figure() -> None:
    """Demonstrate non-rigid warps: a swirl and a sinusoidal wave."""
    img = scene_rgb(max_side=320)
    h, w = img.shape[:2]

    swirled = swirl(img, rotation=0, strength=10, radius=160)

    def wave_map(coords: np.ndarray) -> np.ndarray:
        """Inverse map: each output (col, row) sampled from a sine-shifted row."""
        out = coords.copy()
        out[:, 1] += 12.0 * np.sin(coords[:, 0] / 30.0)
        return out

    waved = warp(img, wave_map, cval=1.0)

    panels = [(img, "Original"),
              (swirled, "Swirl warp (radial twist)"),
              (waved, "Sinusoidal warp (per-row sine shift)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im); ax.set_title(title); ax.axis("off")
    fig.suptitle("Image warping — resampling pixels through a coordinate map",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    savefig(fig, "unit3_warping.png")


def demo() -> None:
    print("[Unit 3a] Image transformations & warping")
    set_style()
    transformations_figure()
    warping_figure()


if __name__ == "__main__":
    demo()
