"""Unit 6 — Motion and change detection.

Covers: dense optical-flow estimation, translational motion estimation by
phase correlation (the "translational alignment" of the syllabus), and change
detection by frame differencing.

Run:  python -m src.classical.unit6_motion
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import shift as nd_shift
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening
from skimage.registration import optical_flow_ilk, phase_cross_correlation
from skimage.transform import rotate

from src.classical.common import rng, savefig, scene_gray, set_style


def _flow_to_rgb(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Encode a flow field as colour: hue = direction, brightness = speed."""
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)
    hsv = np.dstack([(ang + np.pi) / (2 * np.pi),
                     np.ones_like(mag),
                     np.clip(mag / (mag.max() + 1e-8), 0, 1)])
    return hsv_to_rgb(hsv)


# --------------------------------------------------------------------------- #
# 1. Dense optical flow
# --------------------------------------------------------------------------- #
def optical_flow_figure() -> None:
    """Estimate dense optical flow between two frames of a rotating scene."""
    frame1 = scene_gray("camera")
    # Frame 2: the scene rotated slightly — a smooth, rotational motion field.
    frame2 = rotate(frame1, angle=6.0, center=None, mode="edge")

    v, u = optical_flow_ilk(frame1, frame2, radius=7)

    step = 16
    yy, xx = np.mgrid[0:frame1.shape[0]:step, 0:frame1.shape[1]:step]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].imshow(frame1); axes[0].set_title("Frame t")
    axes[1].imshow(frame2); axes[1].set_title("Frame t+1  (rotated 6 deg)")
    axes[2].imshow(_flow_to_rgb(v, u))
    axes[2].quiver(xx, yy, u[::step, ::step], -v[::step, ::step],
                   color="white", scale=60, width=0.003)
    axes[2].set_title("Estimated optical flow\n(colour = direction, arrows = vectors)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Optical flow — recovering per-pixel motion between two frames",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit6_optical_flow.png")


# --------------------------------------------------------------------------- #
# 2. Translational motion estimation
# --------------------------------------------------------------------------- #
def motion_estimation_figure() -> None:
    """Recover a known translation by phase correlation, then align the frames."""
    frame1 = np.clip(scene_gray("astronaut"), 0, 1)
    true_shift = (-17.0, 24.0)                       # (row, col) ground truth
    frame2 = np.clip(nd_shift(frame1, shift=true_shift, mode="reflect"), 0, 1)

    est_shift, error, _ = phase_cross_correlation(frame1, frame2,
                                                  upsample_factor=20)
    # phase_cross_correlation returns the shift that registers the moving
    # frame back onto the reference, so we apply it directly to frame2.
    aligned = np.clip(nd_shift(frame2, shift=est_shift, mode="reflect"), 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].imshow(frame1); axes[0].set_title("Reference frame")
    axes[1].imshow(frame2)
    axes[1].set_title(f"Shifted frame\ntrue shift = {true_shift}")
    # Before alignment: red/green overlay shows the mismatch.
    axes[2].imshow(np.dstack([frame1, frame2, np.zeros_like(frame1)]))
    axes[2].set_title("Overlay BEFORE\n(ghosting = motion)")
    axes[3].imshow(np.dstack([frame1, aligned, np.zeros_like(frame1)]))
    axes[3].set_title(f"Overlay AFTER alignment\nestimated = ({est_shift[0]:.1f}, "
                      f"{est_shift[1]:.1f})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Translational motion estimation by phase correlation",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit6_motion_estimation.png")


# --------------------------------------------------------------------------- #
# 3. Change detection by frame differencing
# --------------------------------------------------------------------------- #
def change_detection_figure() -> None:
    """Detect a moved object against a static background by differencing."""
    background = scene_gray("coffee")
    h, w = background.shape

    def add_object(img, cy, cx, radius=26):
        out = img.copy()
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        out[mask] = 0.05                              # a dark object
        return out

    gen = rng()
    frame1 = add_object(background, h * 0.35, w * 0.30)
    frame1 += gen.normal(0, 0.01, frame1.shape)       # mild sensor noise
    frame2 = add_object(background, h * 0.62, w * 0.68)
    frame2 += gen.normal(0, 0.01, frame2.shape)

    diff = np.abs(frame2 - frame1)
    mask = opening(diff > 0.25, disk(3))              # threshold + clean up

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].imshow(np.clip(frame1, 0, 1)); axes[0].set_title("Frame 1")
    axes[1].imshow(np.clip(frame2, 0, 1)); axes[1].set_title("Frame 2 (object moved)")
    axes[2].imshow(diff, cmap="inferno")
    axes[2].set_title("|Frame2 - Frame1|\nabsolute difference")
    axes[3].imshow(np.clip(frame2, 0, 1))
    for region in regionprops(label(mask)):
        if region.area < 80:
            continue
        minr, minc, maxr, maxc = region.bbox
        axes[3].add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor="lime", linewidth=2))
    axes[3].set_title("Detected change regions")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Change detection — thresholded frame differencing flags motion",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit6_change_detection.png")


def demo() -> None:
    print("[Unit 6] Motion & change detection")
    set_style()
    optical_flow_figure()
    motion_estimation_figure()
    change_detection_figure()


if __name__ == "__main__":
    demo()
