"""Unit 2c — Hough transform and RANSAC.

Covers: the Hough transform for detecting lines and circles by voting in a
parameter space, and RANSAC for fitting a model robustly when a large fraction
of the data are outliers.

Run:  python -m src.classical.unit2_hough_ransac
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line as draw_line
from skimage.feature import canny
from skimage.measure import LineModelND, ransac
from skimage.transform import (hough_circle, hough_circle_peaks, hough_line,
                               hough_line_peaks)

from src.classical.common import coins, rng, savefig, set_style


# --------------------------------------------------------------------------- #
# 1. Hough line transform
# --------------------------------------------------------------------------- #
def _synthetic_lines(n: int = 256) -> np.ndarray:
    """A binary image with a few straight lines plus salt noise."""
    img = np.zeros((n, n), dtype=bool)
    segments = [(20, 20, 230, 200), (200, 10, 40, 240), (10, 130, 245, 130)]
    for r0, c0, r1, c1 in segments:
        rr, cc = draw_line(r0, c0, r1, c1)
        img[rr, cc] = True
    # Sprinkle outlier pixels so the vote-based detector has to cope with noise.
    noise = rng().random((n, n)) < 0.004
    return img | noise


def hough_lines_figure() -> None:
    """Detect straight lines by peak-finding in Hough (angle, distance) space."""
    img = _synthetic_lines()
    h, theta, d = hough_line(img)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].imshow(img); axes[0].set_title("Edge image (lines + noise)")
    axes[0].axis("off")

    axes[1].imshow(np.log1p(h), extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]),
                   d[-1], d[0]], aspect="auto", cmap="hot")
    axes[1].set_title("Hough accumulator\n(each bright spot = one line)")
    axes[1].set_xlabel("angle theta (deg)"); axes[1].set_ylabel("distance rho")

    axes[2].imshow(img, cmap="gray")
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=3)):
        x0, y0 = dist * np.array([np.cos(angle), np.sin(angle)])
        axes[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2), color="red")
    axes[2].set_xlim(0, img.shape[1]); axes[2].set_ylim(img.shape[0], 0)
    axes[2].set_title("Recovered lines (3 peaks)"); axes[2].axis("off")

    fig.suptitle("Hough line transform — detecting lines by voting in parameter space",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit2_hough_lines.png")


# --------------------------------------------------------------------------- #
# 2. Hough circle transform
# --------------------------------------------------------------------------- #
def hough_circles_figure() -> None:
    """Detect coins as circles via the Hough circle transform."""
    img = coins()
    # Quantile thresholds so the detector works regardless of the image's
    # value range (coins() is a float image in [0, 1], not 8-bit [0, 255]).
    edges = canny(img, sigma=2.5, low_threshold=0.7, high_threshold=0.92,
                  use_quantiles=True)

    radii = np.arange(20, 50, 2)
    res = hough_circle(edges, radii)
    accums, cx, cy, cr = hough_circle_peaks(res, radii, total_num_peaks=24,
                                            min_xdistance=20, min_ydistance=20)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(img); axes[0].set_title("Input: coins")
    axes[1].imshow(edges); axes[1].set_title("Canny edges")
    axes[2].imshow(img)
    for x, y, r in zip(cx, cy, cr):
        axes[2].add_patch(plt.Circle((x, y), r, color="red", fill=False, linewidth=1.6))
        axes[2].plot(x, y, "+", color="red")
    axes[2].set_title(f"{len(cx)} circles detected")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Hough circle transform — each edge pixel votes for circle centres",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit2_hough_circles.png")


# --------------------------------------------------------------------------- #
# 3. RANSAC robust fitting
# --------------------------------------------------------------------------- #
def ransac_figure() -> None:
    """Fit a line to outlier-heavy data: least squares vs RANSAC."""
    gen = rng()
    n_in, n_out = 80, 55
    # Inliers along y = 0.6 x + 12 with small noise.
    x_in = np.linspace(0, 100, n_in)
    y_in = 0.6 * x_in + 12 + gen.normal(0, 3.0, n_in)
    # Outliers scattered anywhere.
    x_out = gen.uniform(0, 100, n_out)
    y_out = gen.uniform(0, 100, n_out)
    data = np.column_stack([np.r_[x_in, x_out], np.r_[y_in, y_out]])

    ls = LineModelND.from_estimate(data)                 # plain least squares
    robust, inliers = ransac(data, LineModelND, min_samples=2,
                             residual_threshold=6.0, max_trials=1000,
                             rng=42)
    outliers = ~inliers

    xs = np.array([0, 100])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(data[inliers, 0], data[inliers, 1], ".", color="#2563eb",
            label=f"RANSAC inliers ({inliers.sum()})")
    ax.plot(data[outliers, 0], data[outliers, 1], ".", color="#9ca3af",
            label=f"RANSAC outliers ({outliers.sum()})")
    ax.plot(xs, ls.predict_y(xs), "--", color="crimson", linewidth=2,
            label="Least squares (dragged by outliers)")
    ax.plot(xs, robust.predict_y(xs), "-", color="green", linewidth=2.5,
            label="RANSAC fit (robust)")
    ax.plot(xs, 0.6 * xs + 12, ":", color="black", label="ground truth")
    ax.set_title("RANSAC — robust model fitting under heavy outlier contamination",
                 fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "unit2_ransac_line.png")


def demo() -> None:
    print("[Unit 2c] Hough transform & RANSAC")
    set_style()
    hough_lines_figure()
    hough_circles_figure()
    ransac_figure()


if __name__ == "__main__":
    demo()
