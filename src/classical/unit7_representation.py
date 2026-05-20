"""Unit 7 — Generative models and representation learning.

Covers: PCA as a learned representation (an autoencoder's linear cousin),
2-D embeddings that expose class structure, perceptual grouping by image
segmentation, and a simple *linear generative model* that samples brand-new
images from a learned latent distribution.

This is the conceptual home of VeriVision itself: the project detects images
produced by *conditional generative models* (Stable Diffusion).

Run:  python -m src.classical.unit7_representation
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.segmentation import felzenszwalb, mark_boundaries, quickshift, slic

from src.classical.common import coffee, savefig, set_style


# --------------------------------------------------------------------------- #
# 1. PCA — a learned representation
# --------------------------------------------------------------------------- #
def pca_eigenimages_figure() -> None:
    """Learn 'eigen-digits' with PCA and reconstruct from a few components."""
    digits = load_digits()
    X = digits.data
    pca = PCA(n_components=40, random_state=42).fit(X)

    fig = plt.figure(figsize=(13, 5.6))

    # Row 1: mean image + leading principal components ("eigen-digits").
    ax = fig.add_subplot(2, 6, 1)
    ax.imshow(pca.mean_.reshape(8, 8)); ax.set_title("mean", fontsize=9)
    ax.axis("off")
    for i in range(5):
        ax = fig.add_subplot(2, 6, i + 2)
        ax.imshow(pca.components_[i].reshape(8, 8), cmap="coolwarm")
        ax.set_title(f"PC {i + 1}", fontsize=9); ax.axis("off")

    # Row 2: progressive reconstruction of one digit.
    sample = X[15]
    for j, k in enumerate([1, 2, 8, 20, 40]):
        coeffs = pca.transform(sample[None])[:, :k]
        recon = coeffs @ pca.components_[:k] + pca.mean_
        ax = fig.add_subplot(2, 6, 7 + j)
        ax.imshow(recon.reshape(8, 8)); ax.axis("off")
        ax.set_title(f"{k} comp.", fontsize=9)
    ax = fig.add_subplot(2, 6, 12)
    ax.imshow(sample.reshape(8, 8)); ax.set_title("original", fontsize=9)
    ax.axis("off")

    fig.suptitle("PCA representation learning — eigen-images (top) and "
                 "reconstruction from k components (bottom)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit7_pca_eigenimages.png")


# --------------------------------------------------------------------------- #
# 2. Low-dimensional embeddings
# --------------------------------------------------------------------------- #
def embedding_figure() -> None:
    """Project 64-D digit images to 2-D with PCA and t-SNE."""
    digits = load_digits()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(digits.data), size=700, replace=False)
    X, y = digits.data[idx], digits.target[idx]

    pca_2d = PCA(n_components=2, random_state=42).fit_transform(X)
    tsne_2d = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=42).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, emb, title in [(axes[0], pca_2d, "PCA — linear projection"),
                           (axes[1], tsne_2d, "t-SNE — non-linear embedding")]:
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=14,
                        alpha=0.85)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axes, label="digit class", fraction=0.025)
    fig.suptitle("Representation learning — good features make classes separable",
                 fontweight="bold")
    savefig(fig, "unit7_embedding.png")


# --------------------------------------------------------------------------- #
# 3. Perceptual grouping — image segmentation
# --------------------------------------------------------------------------- #
def segmentation_figure() -> None:
    """Group pixels into perceptual regions with four segmentation methods."""
    img = coffee()

    slic_seg = slic(img, n_segments=250, compactness=12, start_label=1,
                    channel_axis=-1)
    fz_seg = felzenszwalb(img, scale=220, sigma=0.6, min_size=60)
    qs_seg = quickshift(img, kernel_size=5, max_dist=12, ratio=0.5)

    flat = img.reshape(-1, 3)
    km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(flat)
    km_img = km.cluster_centers_[km.labels_].reshape(img.shape)

    panels = [
        (mark_boundaries(img, slic_seg), f"SLIC superpixels\n({slic_seg.max()} regions)"),
        (mark_boundaries(img, fz_seg), f"Felzenszwalb graph cut\n({fz_seg.max() + 1} regions)"),
        (mark_boundaries(img, qs_seg), f"Quickshift\n({qs_seg.max() + 1} regions)"),
        (km_img, "K-means colour clustering\n(6 colours)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im); ax.set_title(title); ax.axis("off")
    fig.suptitle("Perceptual grouping — segmenting an image into coherent regions",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit7_segmentation.png")


# --------------------------------------------------------------------------- #
# 4. A simple generative model
# --------------------------------------------------------------------------- #
def generative_figure() -> None:
    """Sample new images from a Gaussian latent space — a linear generative model.

    Fit PCA, treat the principal-component scores as a latent code with a
    diagonal Gaussian distribution, sample fresh codes, and decode them. This
    is the same encode -> sample latent -> decode recipe behind VAEs; modern
    diffusion models replace the linear decoder with a deep denoiser.
    """
    digits = load_digits()
    threes = digits.data[digits.target == 3]
    n_latent = 8
    pca = PCA(n_components=n_latent, random_state=42).fit(threes)

    scores = pca.transform(threes)
    std = scores.std(axis=0)                          # per-latent spread
    gen = np.random.default_rng(3)
    # Sample latent codes; a temperature < 1 keeps draws near the data manifold.
    temperature = 0.7
    samples = gen.normal(0, 1, (40, n_latent)) * std * temperature
    generated = samples @ pca.components_ + pca.mean_

    fig, axes = plt.subplots(5, 8, figsize=(11, 7))
    for ax, img in zip(axes.ravel(), generated):
        ax.imshow(np.clip(img.reshape(8, 8), 0, 16)); ax.axis("off")
    fig.suptitle("Generative model — 40 brand-new digit '3's sampled from a "
                 "learned latent Gaussian", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit7_generative.png")


def demo() -> None:
    print("[Unit 7] Generative models & representation learning")
    set_style()
    pca_eigenimages_figure()
    embedding_figure()
    segmentation_figure()
    generative_figure()


if __name__ == "__main__":
    demo()
