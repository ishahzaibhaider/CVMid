"""End-to-end demo that regenerates all figures for the report/slides.

Run after training at least one model. Produces:
  reports/figures/
    sample_grid.png          — sample REAL and FAKE images side by side
    fft_comparison.png       — avg FFT spectrum for REAL vs FAKE
    gradcam_grid.png         — Grad-CAM overlays for 6 test images
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CKPT_DIR, DATA_DIR, FIGURES_DIR  # noqa: E402
from src.evaluation.gradcam import gradcam_overlay  # noqa: E402
from src.models import build_model, gradcam_target_layer  # noqa: E402
from src.preprocessing import build_eval_transform  # noqa: E402
from src.preprocessing.frequency import compute_fft_magnitude  # noqa: E402
from src.utils import resolve_device  # noqa: E402


def sample_paths(split: str, cls: str, n: int):
    root = DATA_DIR / split / cls
    if not root.exists():
        raise FileNotFoundError(f"Missing {root}. See data/README.md.")
    files = sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    return files[:n]


def plot_sample_grid(out_path: Path, n_per_class: int = 8) -> None:
    reals = sample_paths("test", "REAL", n_per_class)
    fakes = sample_paths("test", "FAKE", n_per_class)
    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 1.5, 3.5))
    for j, p in enumerate(reals):
        axes[0, j].imshow(Image.open(p))
        axes[0, j].axis("off")
    for j, p in enumerate(fakes):
        axes[1, j].imshow(Image.open(p))
        axes[1, j].axis("off")
    axes[0, 0].set_title("REAL", loc="left")
    axes[1, 0].set_title("FAKE", loc="left")
    fig.suptitle("CIFAKE samples — 32x32 RGB")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fft_comparison(out_path: Path, n: int = 200) -> None:
    reals = sample_paths("test", "REAL", n)
    fakes = sample_paths("test", "FAKE", n)

    def avg_spectrum(paths):
        accum = None
        for p in paths:
            arr = np.asarray(Image.open(p).convert("RGB"))
            s = compute_fft_magnitude(arr)
            accum = s if accum is None else accum + s
        return accum / len(paths)

    spec_real = avg_spectrum(reals)
    spec_fake = avg_spectrum(fakes)
    diff = spec_fake - spec_real

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(spec_real, cmap="magma")
    axes[0].set_title(f"Mean FFT (REAL, n={n})")
    axes[1].imshow(spec_fake, cmap="magma")
    axes[1].set_title(f"Mean FFT (FAKE, n={n})")
    axes[2].imshow(diff, cmap="seismic", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
    axes[2].set_title("FAKE − REAL  (spectral fingerprint)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gradcam_grid(ckpt_path: Path, out_path: Path, n: int = 6) -> None:
    device = resolve_device("auto")
    blob = torch.load(ckpt_path, map_location=device)
    meta = blob.get("meta", {})
    model_name = meta.get("model", "custom_cnn")
    in_channels = meta.get("in_channels", 3)
    use_fft = meta.get("use_fft_channel", False)
    image_size = meta.get("image_size", 32 if model_name == "custom_cnn" else 224)

    model, _ = build_model(model_name, in_channels=in_channels, num_classes=2)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    target_layer = gradcam_target_layer(model, model_name)
    tf = build_eval_transform(model_name, image_size, use_fft)

    reals = sample_paths("test", "REAL", n // 2)
    fakes = sample_paths("test", "FAKE", n // 2)
    paths = list(zip(reals + fakes, ["REAL"] * len(reals) + ["FAKE"] * len(fakes)))

    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))
    for j, (p, true_cls) in enumerate(paths):
        pil = Image.open(p).convert("RGB")
        tensor = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = ["FAKE", "REAL"][pred_idx]
        disp = np.asarray(pil.resize((224, 224))).astype(np.float32) / 255.0
        overlay = gradcam_overlay(model, target_layer, tensor, disp, target_class=pred_idx)
        axes[0, j].imshow(disp)
        axes[0, j].axis("off")
        axes[0, j].set_title(f"true: {true_cls}", fontsize=9)
        axes[1, j].imshow(overlay)
        axes[1, j].axis("off")
        axes[1, j].set_title(
            f"pred: {pred_name} ({probs[pred_idx]:.2f})",
            fontsize=9,
            color="green" if pred_name == true_cls else "red",
        )
    fig.suptitle("Grad-CAM overlays on CIFAKE test set")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("1/3  Sample grid ...")
    plot_sample_grid(FIGURES_DIR / "sample_grid.png")
    print("2/3  FFT spectral comparison ...")
    plot_fft_comparison(FIGURES_DIR / "fft_comparison.png")

    checkpoints = sorted(CKPT_DIR.glob("*.pt"))
    if checkpoints:
        print(f"3/3  Grad-CAM grid (using {checkpoints[0].name}) ...")
        plot_gradcam_grid(checkpoints[0], FIGURES_DIR / "gradcam_grid.png")
    else:
        print("3/3  Skipping Grad-CAM grid — no checkpoints in models_ckpt/. Train a model first.")

    print(f"\nDone. Figures are in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
