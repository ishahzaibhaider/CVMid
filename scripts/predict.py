"""Predict REAL vs FAKE on a single image or a directory of images.

Usage:
    python -m scripts.predict --checkpoint models_ckpt/custom_cnn.pt --image path/to/img.jpg
    python -m scripts.predict --checkpoint models_ckpt/custom_cnn.pt --dir path/to/folder --gradcam
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASS_NAMES, FIGURES_DIR  # noqa: E402
from src.evaluation.gradcam import gradcam_overlay  # noqa: E402
from src.models import build_model, gradcam_target_layer  # noqa: E402
from src.preprocessing import build_eval_transform  # noqa: E402
from src.utils import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to a single image")
    g.add_argument("--dir", type=str, help="Path to a directory of images")
    p.add_argument("--gradcam", action="store_true", help="Save Grad-CAM overlays for each prediction")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def iter_images(args) -> Iterable[Path]:
    if args.image:
        yield Path(args.image)
        return
    root = Path(args.dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    blob = torch.load(args.checkpoint, map_location=device)
    meta = blob.get("meta", {})
    model_name = meta.get("model", "custom_cnn")
    in_channels = meta.get("in_channels", 3)
    use_fft = meta.get("use_fft_channel", False)
    image_size = meta.get("image_size", 32 if model_name == "custom_cnn" else 224)

    model, target_layer = build_model(model_name, in_channels=in_channels, num_classes=2)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    target_layer = gradcam_target_layer(model, model_name)

    tf = build_eval_transform(model_name, image_size, use_fft)

    for img_path in iter_images(args):
        pil = Image.open(img_path).convert("RGB")
        tensor = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        label = CLASS_NAMES[pred_idx]
        print(f"{img_path}  →  {label}  (confidence {conf:.3f}   P[FAKE]={probs[0]:.3f}  P[REAL]={probs[1]:.3f})")

        if args.gradcam:
            disp = np.asarray(pil.resize((224, 224))).astype(np.float32) / 255.0
            out_path = FIGURES_DIR / "gradcam" / f"{img_path.stem}_{label}.png"
            gradcam_overlay(
                model=model,
                target_layer=target_layer,
                input_tensor=tensor,
                original_rgb=disp,
                target_class=pred_idx,
                save_path=out_path,
            )


if __name__ == "__main__":
    main()
