"""Unit 4 — Model fitting, regularisation and kernel regression.

Covers: polynomial curve fitting, the under-fitting / over-fitting trade-off
seen through train-vs-test error, ridge (L2) regularisation as a cure for
over-fitting, and kernel regression as a non-parametric alternative.

These ideas are the statistical backbone of the deep models in Unit 5: a CNN
is just a very expressive fitted model, and weight-decay is ridge regression.

Run:  python -m src.classical.unit4_model_fitting
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from src.classical.common import rng, savefig, set_style

# Ground-truth function the demos try to recover from noisy samples.
_TRUE = lambda x: np.sin(2.0 * np.pi * x)


def _dataset(n: int, noise: float, seed_offset: int = 0):
    """Sample noisy observations of the true sine function on [0, 1]."""
    gen = np.random.default_rng(42 + seed_offset)
    x = np.sort(gen.uniform(0, 1, n))
    y = _TRUE(x) + gen.normal(0, noise, n)
    return x, y


# --------------------------------------------------------------------------- #
# 1. Polynomial curve fitting
# --------------------------------------------------------------------------- #
def polyfit_figure() -> None:
    """Fit polynomials of increasing degree to the same noisy data."""
    x, y = _dataset(n=25, noise=0.18)
    grid = np.linspace(0, 1, 400)
    degrees = [1, 3, 9, 15]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, deg in zip(axes.ravel(), degrees):
        coeffs = np.polyfit(x, y, deg)
        ax.plot(grid, _TRUE(grid), "g-", label="true function")
        ax.scatter(x, y, color="black", s=22, zorder=3, label="noisy samples")
        ax.plot(grid, np.polyval(coeffs, grid), "r-", linewidth=2,
                label=f"degree-{deg} fit")
        verdict = {1: "under-fit", 3: "good fit", 9: "over-fitting",
                   15: "severe over-fit"}[deg]
        ax.set_title(f"Polynomial degree {deg}  —  {verdict}")
        ax.set_ylim(-2, 2); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=8)
    fig.suptitle("Polynomial curve fitting — model capacity vs. the data",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit4_polyfit.png")


# --------------------------------------------------------------------------- #
# 2. The under-fitting / over-fitting trade-off
# --------------------------------------------------------------------------- #
def over_underfitting_figure() -> None:
    """Plot train vs test error against model complexity (polynomial degree).

    Errors are averaged over many fresh training sets so the curves show the
    expected bias-variance behaviour rather than one noisy draw.
    """
    x_te, y_te = _dataset(n=400, noise=0.18, seed_offset=999)
    degrees = list(range(1, 15))
    trials = 80
    rmse = lambda a, b: np.sqrt(np.mean((a - b) ** 2))

    train_err, test_err = [], []
    for deg in degrees:
        tr, te = [], []
        for k in range(trials):
            xk, yk = _dataset(n=25, noise=0.18, seed_offset=k + 1)
            coeffs = np.polyfit(xk, yk, deg)
            tr.append(rmse(np.polyval(coeffs, xk), yk))
            te.append(rmse(np.polyval(coeffs, x_te), y_te))
        train_err.append(float(np.mean(tr)))
        test_err.append(float(np.median(te)))   # median is robust to rare blow-ups

    best = degrees[int(np.argmin(test_err))]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(degrees, train_err, "o-", color="#2563eb", label="training error")
    ax.plot(degrees, test_err, "s-", color="crimson", label="test error")
    ax.axvline(best, color="green", linestyle="--",
               label=f"best generalisation (deg {best})")
    ax.fill_between(degrees, 0, 1, where=[d < best for d in degrees],
                    color="orange", alpha=0.08)
    ax.fill_between(degrees, 0, 1, where=[d > best for d in degrees],
                    color="purple", alpha=0.08)
    ax.text(2, 0.62, "UNDER-FITTING\n(high bias)", fontsize=9, color="darkorange")
    ax.text(11.4, 0.62, "OVER-FITTING\n(high variance)", fontsize=9, color="purple")
    ax.set_title("Bias-variance trade-off — train error falls, test error is U-shaped",
                 fontweight="bold")
    ax.set_xlabel("polynomial degree (model complexity)")
    ax.set_ylabel("RMSE"); ax.set_ylim(0, 0.7); ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "unit4_over_underfitting.png")


# --------------------------------------------------------------------------- #
# 3. Regularisation
# --------------------------------------------------------------------------- #
def regularisation_figure() -> None:
    """Tame an over-capacity degree-15 model with ridge (L2) regularisation."""
    x, y = _dataset(n=25, noise=0.18)
    grid = np.linspace(0, 1, 400)
    alphas = [0.0, 1e-6, 1e-3, 1e-1]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, alpha in zip(axes.ravel(), alphas):
        model = make_pipeline(
            PolynomialFeatures(15),
            Ridge(alpha=alpha) if alpha > 0 else LinearRegression(),
        )
        model.fit(x[:, None], y)
        ax.plot(grid, _TRUE(grid), "g-", label="true function")
        ax.scatter(x, y, color="black", s=22, zorder=3)
        ax.plot(grid, model.predict(grid[:, None]), "r-", linewidth=2)
        label = "no regularisation" if alpha == 0 else f"ridge  lambda = {alpha:g}"
        ax.set_title(f"Degree-15 model  —  {label}")
        ax.set_ylim(-2, 2); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(fontsize=8)
    fig.suptitle("Regularisation — an L2 penalty shrinks weights and removes the wiggle",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit4_regularization.png")


# --------------------------------------------------------------------------- #
# 4. Kernel regression
# --------------------------------------------------------------------------- #
def kernel_regression_figure() -> None:
    """Non-parametric fitting: RBF kernel regression at three bandwidths."""
    x, y = _dataset(n=25, noise=0.18)
    grid = np.linspace(0, 1, 400)
    gammas = [2.0, 25.0, 400.0]
    labels = ["wide kernel (under-fit)", "balanced kernel",
              "narrow kernel (over-fit)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, gamma, label in zip(axes, gammas, labels):
        model = KernelRidge(kernel="rbf", alpha=1e-2, gamma=gamma)
        model.fit(x[:, None], y)
        ax.plot(grid, _TRUE(grid), "g-", label="true function")
        ax.scatter(x, y, color="black", s=20, zorder=3)
        ax.plot(grid, model.predict(grid[:, None]), "r-", linewidth=2)
        ax.set_title(f"RBF kernel, gamma = {gamma:g}\n{label}")
        ax.set_ylim(-2, 2); ax.set_xlabel("x"); ax.legend(fontsize=8)
    fig.suptitle("Kernel regression — a non-parametric fit; bandwidth sets smoothness",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, "unit4_kernel_regression.png")


def demo() -> None:
    print("[Unit 4] Model fitting, regularisation & kernel regression")
    set_style()
    polyfit_figure()
    over_underfitting_figure()
    regularisation_figure()
    kernel_regression_figure()


if __name__ == "__main__":
    demo()
