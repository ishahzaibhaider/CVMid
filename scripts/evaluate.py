"""Evaluate a trained VeriVision model on the CIFAKE test set.

Usage:
    python -m scripts.evaluate --checkpoint models_ckpt/custom_cnn.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CKPT_DIR, DATA_DIR, FIGURES_DIR, REPORTS_DIR  # noqa: E402
from src.data import build_dataloaders  # noqa: E402
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix, plot_roc  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import build_eval_transform, build_train_transform  # noqa: E402
from src.utils import load_checkpoint, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = resolve_device(args.device)
    blob = torch.load(ckpt_path, map_location=device)
    meta = blob.get("meta", {})
    model_name = meta.get("model", "custom_cnn")
    in_channels = meta.get("in_channels", 3)
    use_fft = meta.get("use_fft_channel", False)
    image_size = meta.get("image_size", 32 if model_name == "custom_cnn" else 224)
    print(f"Loading {model_name} | image_size={image_size} | in_channels={in_channels} | fft={use_fft}")

    model, _ = build_model(model_name, in_channels=in_channels, num_classes=2)
    model.load_state_dict(blob["state_dict"])
    model.to(device)

    train_tf = build_train_transform(model_name, image_size, use_fft)  # unused but needed for DL
    eval_tf = build_eval_transform(model_name, image_size, use_fft)
    _, _, test_loader = build_dataloaders(
        root=DATA_DIR,
        train_transform=train_tf,
        eval_transform=eval_tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.1,
        seed=42,
    )

    metrics = evaluate_model(model, test_loader, device)

    print("\n=== Test metrics ===")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:<10} {metrics[k]:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred) FAKE/REAL:")
    print(metrics["confusion_matrix"])
    print("\nClassification report:\n" + metrics["classification_report"])

    suffix = f"{model_name}{'_fft' if use_fft else ''}"
    import numpy as np  # local import to keep top imports tidy
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), FIGURES_DIR / f"confusion_{suffix}.png")
    plot_roc(metrics["labels"], metrics["probs"], FIGURES_DIR / f"roc_{suffix}.png")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = REPORTS_DIR / f"metrics_{suffix}.json"
    with result_path.open("w") as f:
        json.dump(
            {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")},
            f,
            indent=2,
        )
    print(f"\nSaved → {result_path}")


if __name__ == "__main__":
    main()
