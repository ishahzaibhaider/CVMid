"""Unit 3b — Stereo vision, depth from disparity, epipolar geometry.

Covers: a rectified stereo pair, a block-matching disparity estimator, the
inverse relationship between disparity and depth, and recovery of the epipolar
geometry (fundamental matrix + epipolar lines) from feature matches.

Run:  python -m src.classical.unit3_stereo
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, rescale

from src.classical.common import savefig, set_style


def _stereo_pair():
    """Return the rectified 'motorcycle' stereo pair (left, right) as float gray."""
    left, right, _ = data.stereo_motorcycle()
    return rgb2gray(left), rgb2gray(right)


# --------------------------------------------------------------------------- #
# 1. The stereo pair
# --------------------------------------------------------------------------- #
def stereo_pair_figure() -> None:
    """Show the left/right views and a red-cyan anaglyph of the pair."""
    left, right = _stereo_pair()
    anaglyph = np.dstack([left, right, right])  # red = left eye, cyan = right eye

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].imshow(left); axes[0].set_title("Left camera view")
    axes[1].imshow(right); axes[1].set_title("Right camera view")
    axes[2].imshow(np.clip(anaglyph, 0, 1))
    axes[2].set_title("Red-cyan anaglyph\n(horizontal shift = disparity)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Stereo vision — two horizontally displaced views of one scene",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit3_stereo_pair.png")


# --------------------------------------------------------------------------- #
# 2. Disparity by block matching  ->  depth
# --------------------------------------------------------------------------- #
def block_matching_disparity(left: np.ndarray, right: np.ndarray,
                             max_disp: int = 64, window: int = 7) -> np.ndarray:
    """Winner-take-all SSD block matching.

    For every candidate disparity d we shift the right image right by d, take
    the squared difference from the left image, and box-filter it into a
    windowed matching cost. The per-pixel argmin over d is the disparity.
    """
    h, w = left.shape
    cost = np.full((max_disp, h, w), np.inf)
    for d in range(max_disp):
        shifted = right.copy()
        if d:
            shifted[:, d:] = right[:, :w - d]
        diff = (left - shifted) ** 2
        # Columns with no real correspondent get a high penalty so they are
        # never chosen as the winner near the left border.
        if d:
            diff[:, :d] = 1.0
        cost[d] = uniform_filter(diff, size=window)
    disp = np.argmin(cost, axis=0).astype(float)
    # A light median filter removes isolated mismatched pixels.
    return median_filter(disp, size=5)


def disparity_figure() -> None:
    """Estimate a disparity map and convert it to a relative depth map."""
    left, right = _stereo_pair()
    # Down-scale for a fast pure-NumPy block match.
    scale = 0.45
    l_s = rescale(left, scale, anti_aliasing=True)
    r_s = rescale(right, scale, anti_aliasing=True)

    disp = block_matching_disparity(l_s, r_s, max_disp=64, window=7)
    # Depth is inversely proportional to disparity:  Z = f * B / d.
    depth = 1.0 / (disp + 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].imshow(l_s); axes[0].set_title("Left view (input)")
    im1 = axes[1].imshow(disp, cmap="plasma")
    axes[1].set_title("Disparity map\n(bright = large shift = near)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(depth, cmap="viridis")
    axes[2].set_title("Relative depth  Z ~ 1/disparity")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Depth from disparity — block matching on a rectified stereo pair",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit3_disparity.png")


# --------------------------------------------------------------------------- #
# 3. Epipolar geometry
# --------------------------------------------------------------------------- #
def epipolar_figure() -> None:
    """Estimate the fundamental matrix from ORB matches and draw epipolar lines."""
    left, right = _stereo_pair()

    orb_l = ORB(n_keypoints=500, fast_threshold=0.05)
    orb_l.detect_and_extract(left)
    orb_r = ORB(n_keypoints=500, fast_threshold=0.05)
    orb_r.detect_and_extract(right)
    matches = match_descriptors(orb_l.descriptors, orb_r.descriptors,
                                cross_check=True)
    # skimage keypoints are (row, col); the transform wants (x, y) = (col, row).
    pts_l = orb_l.keypoints[matches[:, 0]][:, ::-1]
    pts_r = orb_r.keypoints[matches[:, 1]][:, ::-1]

    model, inliers = ransac((pts_l, pts_r), FundamentalMatrixTransform,
                            min_samples=8, residual_threshold=1.0,
                            max_trials=5000, rng=42)
    F = model.params

    in_l, in_r = pts_l[inliers], pts_r[inliers]
    pick = np.linspace(0, len(in_l) - 1, 8).astype(int)
    colours = plt.cm.rainbow(np.linspace(0, 1, len(pick)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    axes[0].imshow(left); axes[0].set_title("Left view — points + epipolar lines")
    axes[1].imshow(right); axes[1].set_title("Right view — points + epipolar lines")
    w = left.shape[1]
    for k, c in zip(pick, colours):
        xl, yl = in_l[k]
        xr, yr = in_r[k]
        # Epiline in the right image for the left point:  l' = F x.
        a, b, cc = F @ np.array([xl, yl, 1.0])
        ys = np.array([(-cc - a * 0) / b, (-cc - a * w) / b])
        axes[1].plot([0, w], ys, color=c, linewidth=1)
        axes[1].plot(xr, yr, "o", color=c, markersize=5, mec="white")
        # Epiline in the left image for the right point:  l = F^T x'.
        a, b, cc = F.T @ np.array([xr, yr, 1.0])
        ys = np.array([(-cc - a * 0) / b, (-cc - a * w) / b])
        axes[0].plot([0, w], ys, color=c, linewidth=1)
        axes[0].plot(xl, yl, "o", color=c, markersize=5, mec="white")
    for ax in axes:
        ax.set_xlim(0, w); ax.axis("off")
    fig.suptitle(f"Epipolar geometry — fundamental matrix from {inliers.sum()} "
                 f"RANSAC inliers (lines are near-horizontal: pair is rectified)",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit3_epipolar.png")


def demo() -> None:
    print("[Unit 3b] Stereo vision, disparity & epipolar geometry")
    set_style()
    stereo_pair_figure()
    disparity_figure()
    epipolar_figure()


if __name__ == "__main__":
    demo()
