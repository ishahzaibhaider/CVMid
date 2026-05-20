"""Unit 1 — Introduction to Computer Vision.

Covers: image formation (the pinhole camera model), the digital image as a
sampled grid of intensities, colour-space representations, and the canonical
computer-vision pipeline that the rest of VeriVision is built on.

Run:  python -m src.classical.unit1_image_basics
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from skimage.color import rgb2gray, rgb2hsv, rgb2lab

from src.classical.common import savefig, scene_rgb, set_style

# --------------------------------------------------------------------------- #
# 1. Image formation — the pinhole camera model
# --------------------------------------------------------------------------- #
def _cube(side: float = 1.0) -> np.ndarray:
    """Return the 8 corner points of an axis-aligned cube (3 x 8)."""
    r = side / 2.0
    corners = np.array(
        [[x, y, z] for x in (-r, r) for y in (-r, r) for z in (-r, r)]
    ).T
    return corners


_CUBE_EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
]


def project_points(points_3d: np.ndarray, K: np.ndarray,
                    R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pinhole projection: x ~ K [R | t] X.

    ``points_3d`` is 3 x N in world coordinates. Returns 2 x N pixel coords.
    This is the exact equation behind every camera — VeriVision's REAL images
    are formed this way, which is why their statistics differ from synthesised
    images that never pass through a physical lens.
    """
    cam = R @ points_3d + t.reshape(3, 1)      # world -> camera frame
    img = K @ cam                              # camera -> image plane
    return img[:2] / img[2]                    # perspective divide


def image_formation_figure() -> None:
    """Show the 3D scene and the perspective image the camera records."""
    cube = _cube(1.4)

    # Camera looking down the +Z axis at a cube placed 4 units away.
    f, cx, cy = 600.0, 320.0, 320.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=float)
    theta = np.deg2rad(28.0)
    R = np.array([[np.cos(theta), 0, np.sin(theta)],
                  [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    t = np.array([0.0, 0.0, 4.0])
    proj = project_points(cube, K, R, t)

    fig = plt.figure(figsize=(11, 4.6))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    for i, j in _CUBE_EDGES:
        ax3d.plot(*[[cube[k, i], cube[k, j]] for k in range(3)], color="#1f77b4")
    ax3d.scatter([0], [0], [0], color="crimson", s=60, label="camera centre")
    for axis, colour in zip(np.eye(3), ("crimson", "green", "navy")):
        ax3d.quiver(0, 0, 0, *axis, length=1.0, color=colour, linewidth=2)
    ax3d.set_title("3D world scene")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper left", fontsize=8)

    ax2d = fig.add_subplot(1, 2, 2)
    for i, j in _CUBE_EDGES:
        ax2d.plot([proj[0, i], proj[0, j]], [proj[1, i], proj[1, j]], color="#1f77b4")
    ax2d.scatter(proj[0], proj[1], color="crimson", s=20, zorder=3)
    ax2d.add_patch(plt.Rectangle((0, 0), 640, 640, fill=False, ec="black"))
    ax2d.set_xlim(0, 640); ax2d.set_ylim(640, 0)
    ax2d.set_aspect("equal")
    ax2d.set_title("2D image  (x = K [R|t] X)")
    ax2d.set_xlabel("pixel u"); ax2d.set_ylabel("pixel v")

    fig.suptitle("Image formation — perspective projection through a pinhole camera",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit1_image_formation.png")


# --------------------------------------------------------------------------- #
# 2. The digital image — channels and intensity histogram
# --------------------------------------------------------------------------- #
def channels_histogram_figure() -> None:
    """Decompose an image into RGB channels and plot its intensity histogram."""
    img = scene_rgb()
    gray = rgb2gray(img)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes[0, 0].imshow(img); axes[0, 0].set_title("RGB image")
    axes[0, 1].imshow(gray); axes[0, 1].set_title("Grayscale (luminance)")

    axes[0, 2].set_title("Per-channel intensity histogram")
    for c, colour in enumerate(("red", "green", "blue")):
        axes[0, 2].hist(img[..., c].ravel(), bins=64, range=(0, 1),
                        color=colour, alpha=0.5, label=colour)
    axes[0, 2].legend(fontsize=8); axes[0, 2].set_xlabel("intensity")

    for c, (name, cmap) in enumerate(
        [("Red channel", "Reds"), ("Green channel", "Greens"), ("Blue channel", "Blues")]
    ):
        axes[1, c].imshow(img[..., c], cmap=cmap)
        axes[1, c].set_title(name)

    for ax in axes.ravel():
        if ax is not axes[0, 2]:
            ax.axis("off")
    fig.suptitle("A digital image is a sampled grid of intensities",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit1_channels_histogram.png")


# --------------------------------------------------------------------------- #
# 3. Colour spaces
# --------------------------------------------------------------------------- #
def colour_spaces_figure() -> None:
    """Compare RGB, HSV and CIELAB representations of the same image."""
    img = scene_rgb()
    hsv = rgb2hsv(img)
    lab = rgb2lab(img)
    lab_show = (lab - lab.min()) / (lab.max() - lab.min())

    panels = [
        (img, "RGB — display space"),
        (hsv[..., 0], "HSV — Hue"),
        (hsv[..., 1], "HSV — Saturation"),
        (hsv[..., 2], "HSV — Value"),
        (lab_show, "CIELAB — perceptual"),
        (lab[..., 0], "CIELAB — Lightness L*"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="viridis" if im.ndim == 2 else None)
        ax.set_title(title); ax.axis("off")
    fig.suptitle("Colour-space representations — each exposes different cues",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit1_color_spaces.png")


# --------------------------------------------------------------------------- #
# 4. The computer-vision pipeline
# --------------------------------------------------------------------------- #
def cv_pipeline_figure() -> None:
    """Draw the end-to-end CV pipeline that VeriVision implements."""
    stages = [
        ("Image\nAcquisition", "camera / dataset\n(CIFAKE)"),
        ("Pre-\nprocessing", "resize, normalise,\nFFT channel"),
        ("Feature\nExtraction", "edges, corners,\nCNN feature maps"),
        ("Model /\nInference", "custom CNN,\nResNet-50"),
        ("Decision /\nOutput", "REAL vs FAKE\n+ Grad-CAM"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.4))
    colours = ["#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6"]
    box_w, box_h, gap = 1.9, 1.5, 0.55
    for i, ((title, detail), colour) in enumerate(zip(stages, colours)):
        x = i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), box_w, box_h,
                                    boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=colour, edgecolor="#1e3a8a"))
        ax.text(x + box_w / 2, 0.95, title, ha="center", va="center",
                fontweight="bold", fontsize=10)
        ax.text(x + box_w / 2, 0.42, detail, ha="center", va="center", fontsize=7.5)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + box_w, box_h / 2), (x + box_w + gap, box_h / 2),
                arrowstyle="-|>", mutation_scale=18, color="#1e3a8a"))
    ax.set_xlim(-0.3, len(stages) * (box_w + gap)); ax.set_ylim(-0.3, box_h + 0.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("The computer-vision pipeline (as realised in VeriVision)",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    savefig(fig, "unit1_cv_pipeline.png")


def demo() -> None:
    print("[Unit 1] Introduction to Computer Vision")
    set_style()
    image_formation_figure()
    channels_histogram_figure()
    colour_spaces_figure()
    cv_pipeline_figure()


if __name__ == "__main__":
    demo()
