"""Unit 2b — Image features: edges, corners, blobs and keypoints.

Covers: gradient & Canny edge detection, Harris corner detection, multi-scale
blob detection (LoG / DoG / DoH), and ORB local features matched across two
views — the building blocks every higher-level CV task relies on.

Run:  python -m src.classical.unit2_features
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import (ORB, blob_dog, blob_doh, blob_log, canny,
                             corner_harris, corner_peaks, match_descriptors)
from skimage.filters import sobel, sobel_h, sobel_v
from skimage.transform import AffineTransform, warp

from src.classical.common import coins, savefig, scene_gray, set_style


# --------------------------------------------------------------------------- #
# 1. Edge detection
# --------------------------------------------------------------------------- #
def edges_figure() -> None:
    """Compare raw gradients with the full Canny edge detector."""
    img = scene_gray("camera")
    gx, gy = sobel_v(img), sobel_h(img)
    magnitude = sobel(img)
    edges = canny(img, sigma=2.0)

    panels = [
        (img, "Input image", "gray"),
        (gx, "Sobel  d/dx  (vertical edges)", "coolwarm"),
        (gy, "Sobel  d/dy  (horizontal edges)", "coolwarm"),
        (magnitude, "Gradient magnitude", "magma"),
        (edges, "Canny edges\n(NMS + hysteresis)", "gray"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (im, title, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap); ax.set_title(title); ax.axis("off")
    fig.suptitle("Edge detection — intensity gradients to clean Canny contours",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit2_edges.png")


# --------------------------------------------------------------------------- #
# 2. Corner detection
# --------------------------------------------------------------------------- #
def corners_figure() -> None:
    """Detect Harris corners — points where intensity changes in all directions."""
    img = scene_gray("astronaut")
    response = corner_harris(img, k=0.05, sigma=1.5)
    corners = corner_peaks(response, min_distance=6, threshold_rel=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img); axes[0].set_title("Input image")
    axes[1].imshow(response, cmap="inferno")
    axes[1].set_title("Harris response R\n(high where corner-like)")
    axes[2].imshow(img)
    axes[2].scatter(corners[:, 1], corners[:, 0], s=28, facecolors="none",
                    edgecolors="lime", linewidths=1.4)
    axes[2].set_title(f"{len(corners)} detected corners")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Harris corner detection — stable, repeatable interest points",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit2_corners.png")


# --------------------------------------------------------------------------- #
# 3. Blob detection
# --------------------------------------------------------------------------- #
def blobs_figure() -> None:
    """Multi-scale blob detection: Laplacian/Difference/Determinant of Gaussian."""
    img = coins()

    detectors = [
        ("Laplacian of Gaussian (LoG)", blob_log(img, max_sigma=30, num_sigma=10,
                                                 threshold=0.12), "cyan"),
        ("Difference of Gaussian (DoG)", blob_dog(img, max_sigma=30, threshold=0.12),
         "yellow"),
        ("Determinant of Hessian (DoH)", blob_doh(img, max_sigma=30, threshold=0.008),
         "magenta"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, (title, blobs, colour) in zip(axes, detectors):
        ax.imshow(img)
        for y, x, sigma in blobs:
            # LoG/DoG report sigma; blob radius ~ sqrt(2)*sigma.
            radius = sigma * np.sqrt(2)
            ax.add_patch(plt.Circle((x, y), radius, color=colour, fill=False,
                                    linewidth=1.5))
        ax.set_title(f"{title}\n{len(blobs)} blobs"); ax.axis("off")
    fig.suptitle("Blob detection — locating circular structures across scales",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit2_blobs.png")


# --------------------------------------------------------------------------- #
# 4. ORB keypoints and descriptor matching
# --------------------------------------------------------------------------- #
def keypoints_figure() -> None:
    """Detect ORB keypoints in two views and match them by descriptor."""
    img1 = scene_gray("astronaut")
    # A second 'view': rotate + scale + translate the same scene.
    tf = AffineTransform(scale=0.8, rotation=np.deg2rad(20), translation=(40, -25))
    img2 = warp(img1, tf.inverse, output_shape=img1.shape, cval=0.0)

    orb1 = ORB(n_keypoints=300, fast_threshold=0.05)
    orb1.detect_and_extract(img1)
    orb2 = ORB(n_keypoints=300, fast_threshold=0.05)
    orb2.detect_and_extract(img2)
    matches = match_descriptors(orb1.descriptors, orb2.descriptors,
                                cross_check=True, max_ratio=0.8)

    # Lay the two images side by side and draw match lines ourselves.
    h = max(img1.shape[0], img2.shape[0])
    w1, w2 = img1.shape[1], img2.shape[1]
    canvas = np.zeros((h, w1 + w2))
    canvas[:img1.shape[0], :w1] = img1
    canvas[:img2.shape[0], w1:w1 + w2] = img2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(canvas)
    kp1, kp2 = orb1.keypoints, orb2.keypoints
    for i, j in matches[:60]:
        y1, x1 = kp1[i]
        y2, x2 = kp2[j]
        ax.plot([x1, x2 + w1], [y1, y2], "-", color="lime", linewidth=0.6)
        ax.plot(x1, y1, ".", color="red", markersize=3)
        ax.plot(x2 + w1, y2, ".", color="red", markersize=3)
    ax.set_title(f"ORB keypoints — {len(matches)} cross-checked matches "
                 f"between two views of the same scene", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    savefig(fig, "unit2_keypoints.png")


def demo() -> None:
    print("[Unit 2b] Features: edges, corners, blobs, keypoints")
    set_style()
    edges_figure()
    corners_figure()
    blobs_figure()
    keypoints_figure()


if __name__ == "__main__":
    demo()
