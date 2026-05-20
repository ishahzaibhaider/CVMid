"""Unit 5 — Learning in Computer Vision.

The deep models that classify CIFAKE (custom CNN, ResNet-50) live in
``src/models`` and are trained by ``scripts/train.py``. This module makes the
*mechanics* behind them concrete and runnable without a GPU:

* convolution kernels — the single operation a CNN layer is built from;
* optimisers — SGD vs SGD+momentum vs Adam descending a loss surface;
* a from-scratch 2-layer neural network trained by back-propagation.

Run:  python -m src.classical.unit5_learning
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from sklearn.datasets import make_moons

from src.classical.common import savefig, scene_gray, set_style


# --------------------------------------------------------------------------- #
# 1. Convolution kernels — the CNN building block
# --------------------------------------------------------------------------- #
def convolution_figure() -> None:
    """Apply hand-designed kernels — exactly what a CNN layer learns to do."""
    img = scene_gray("camera")
    kernels = {
        "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], float),
        "Box blur": np.ones((3, 3)) / 9.0,
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
        "Sobel  d/dx": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float),
        "Sobel  d/dy": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float),
        "Laplacian": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
        "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
        "Outline": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], float),
    }
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (name, k) in zip(axes.ravel(), kernels.items()):
        out = convolve(img, k, mode="reflect")
        signed = name in ("Sobel  d/dx", "Sobel  d/dy", "Laplacian", "Emboss")
        ax.imshow(out, cmap="coolwarm" if signed else "gray")
        ax.set_title(name); ax.axis("off")
    fig.suptitle("Convolution kernels — a CNN layer is a stack of *learned* kernels",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit5_convolution.png")


# --------------------------------------------------------------------------- #
# 2. Optimisers — descending a loss surface
# --------------------------------------------------------------------------- #
# An ill-conditioned quadratic bowl: the y-axis is 60x steeper than the x-axis,
# so plain gradient descent is forced into a tiny step size and crawls.
_COND = 60.0
# Each optimiser gets its own well-tuned (lr, momentum) pair.
_HYPER = {"SGD": (1.6 / _COND, 0.0), "Momentum": (1.0 / _COND, 0.85),
          "Adam": (0.25, 0.0)}
_STEPS = 90
_START = np.array([9.0, 2.5])


def _loss(p):
    x, y = p
    return 0.5 * (x ** 2 + _COND * y ** 2)


def _grad(p):
    x, y = p
    return np.array([x, _COND * y])


def _run_optimiser(kind: str):
    """Return the descent trajectory of one optimiser from a fixed start."""
    lr, beta = _HYPER[kind]
    p = _START.copy()
    path = [p.copy()]
    v = np.zeros(2)          # momentum / 1st moment
    s = np.zeros(2)          # Adam 2nd moment
    for t in range(1, _STEPS + 1):
        g = _grad(p)
        if kind == "SGD":
            p = p - lr * g
        elif kind == "Momentum":
            v = beta * v - lr * g
            p = p + v
        elif kind == "Adam":
            v = 0.9 * v + 0.1 * g
            s = 0.999 * s + 0.001 * g ** 2
            vh, sh = v / (1 - 0.9 ** t), s / (1 - 0.999 ** t)
            p = p - lr * vh / (np.sqrt(sh) + 1e-8)
        path.append(p.copy())
    return np.array(path)


def optimisers_figure() -> None:
    """Compare SGD, SGD+momentum and Adam on the same loss surface."""
    paths = {k: _run_optimiser(k) for k in ("SGD", "Momentum", "Adam")}
    colours = {"SGD": "crimson", "Momentum": "darkorange", "Adam": "#2563eb"}

    gx, gy = np.meshgrid(np.linspace(-10, 10, 240), np.linspace(-3, 3, 240))
    z = 0.5 * (gx ** 2 + _COND * gy ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].contour(gx, gy, z, levels=np.logspace(0, 2.7, 22), cmap="Greys",
                    linewidths=0.6)
    for name, path in paths.items():
        axes[0].plot(path[:, 0], path[:, 1], "o-", color=colours[name],
                     markersize=3, linewidth=1.3, label=name)
    axes[0].plot(0, 0, "*", color="green", markersize=16, label="minimum")
    axes[0].set_title("Descent trajectories on an ill-conditioned loss surface")
    axes[0].set_xlabel("weight 1"); axes[0].set_ylabel("weight 2")
    axes[0].legend(fontsize=8)

    for name, path in paths.items():
        losses = [_loss(p) for p in path]
        axes[1].semilogy(losses, color=colours[name], linewidth=1.8, label=name)
    axes[1].set_title("Loss vs. iteration (log scale)")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("loss")
    axes[1].legend(fontsize=8)
    fig.suptitle("Optimisers — momentum and Adam beat plain gradient descent",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit5_optimizers.png")


# --------------------------------------------------------------------------- #
# 3. A neural network trained from scratch by back-propagation
# --------------------------------------------------------------------------- #
def _train_mlp(X, y, hidden=16, epochs=4000, lr=0.5, l2=1e-3, seed=42):
    """A 2-layer MLP (ReLU hidden, sigmoid output) — full forward/backward by hand."""
    gen = np.random.default_rng(seed)
    W1 = gen.normal(0, 0.5, (X.shape[1], hidden))
    b1 = np.zeros(hidden)
    W2 = gen.normal(0, 0.5, (hidden, 1))
    b2 = np.zeros(1)
    yc = y.reshape(-1, 1).astype(float)
    losses = []
    for _ in range(epochs):
        # ---- forward ----
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)                       # ReLU
        z2 = a1 @ W2 + b2
        out = 1.0 / (1.0 + np.exp(-z2))              # sigmoid
        eps = 1e-9
        bce = -np.mean(yc * np.log(out + eps) + (1 - yc) * np.log(1 - out + eps))
        losses.append(bce + l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2)))
        # ---- backward (chain rule) ----
        n = len(X)
        dz2 = (out - yc) / n
        dW2 = a1.T @ dz2 + 2 * l2 * W2
        db2 = dz2.sum(0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1 + 2 * l2 * W1
        db1 = dz1.sum(0)
        # ---- SGD update ----
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2
    predict = lambda Z: 1.0 / (1.0 + np.exp(-(np.maximum(0, Z @ W1 + b1) @ W2 + b2)))
    return predict, losses


def neural_network_figure() -> None:
    """Train a from-scratch MLP on a non-linearly-separable dataset."""
    X, y = make_moons(n_samples=320, noise=0.22, random_state=42)
    predict, losses = _train_mlp(X, y)

    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 300),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 300))
    prob = predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    acc = float(np.mean((predict(X)[:, 0] > 0.5) == y))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].contourf(gx, gy, prob, levels=20, cmap="RdBu", alpha=0.8)
    axes[0].contour(gx, gy, prob, levels=[0.5], colors="black", linewidths=1.5)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors="k", s=25)
    axes[0].set_title(f"Learned non-linear decision boundary  (train acc {acc:.1%})")
    axes[0].set_xlabel("feature 1"); axes[0].set_ylabel("feature 2")

    axes[1].plot(losses, color="#2563eb", linewidth=1.8)
    axes[1].set_title("Training loss — back-propagation + SGD")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("binary cross-entropy + L2")
    fig.suptitle("A neural network learned from scratch (forward, back-prop, SGD)",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit5_neural_network.png")


def demo() -> None:
    print("[Unit 5] Learning in Computer Vision")
    set_style()
    convolution_figure()
    optimisers_figure()
    neural_network_figure()


if __name__ == "__main__":
    demo()
