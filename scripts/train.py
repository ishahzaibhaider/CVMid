"""Train a VeriVision model on CIFAKE.

Usage:
    python -m scripts.train --model custom_cnn --epochs 10
    python -m scripts.train --model resnet50 --epochs 5 --batch-size 64
    python -m scripts.train --model custom_cnn --use-fft-channel --epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` too
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CKPT_DIR, DATA_DIR, FIGURES_DIR, TrainConfig  # noqa: E402
from src.data import build_dataloaders  # noqa: E402
from src.evaluation.metrics import plot_training_curves  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import build_eval_transform, build_train_transform  # noqa: E402
from src.training import Trainer  # noqa: E402
from src.utils import resolve_device, set_seed  # noqa: E402


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train VeriVision model on CIFAKE")
    p.add_argument("--model", choices=["custom_cnn", "resnet50"], default="custom_cnn")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-fft-channel", action="store_true")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--checkpoint-name", type=str, default="")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    ckpt_name = args.checkpoint_name or (
        f"{args.model}{'_fft' if args.use_fft_channel else ''}.pt"
    )
    return TrainConfig(
        model=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        use_fft_channel=args.use_fft_channel,
        image_size=args.image_size,
        val_split=args.val_split,
        early_stop_patience=args.patience,
        checkpoint_name=ckpt_name,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    image_size = cfg.resolved_image_size()
    print(f"Device: {device}  |  image_size: {image_size}  |  model: {cfg.model}  |  fft: {cfg.use_fft_channel}")

    train_tf = build_train_transform(cfg.model, image_size, cfg.use_fft_channel)
    eval_tf = build_eval_transform(cfg.model, image_size, cfg.use_fft_channel)
    train_loader, val_loader, _ = build_dataloaders(
        root=DATA_DIR,
        train_transform=train_tf,
        eval_transform=eval_tf,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    in_channels = 4 if cfg.use_fft_channel else 3
    model, _ = build_model(cfg.model, in_channels=in_channels, num_classes=2)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    ckpt_path = CKPT_DIR / cfg.checkpoint_name
    meta = {
        "model": cfg.model,
        "use_fft_channel": cfg.use_fft_channel,
        "image_size": image_size,
        "in_channels": in_channels,
    }
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        early_stop_patience=cfg.early_stop_patience,
        checkpoint_path=ckpt_path,
        meta=meta,
    )
    history = trainer.fit()

    curves_path = FIGURES_DIR / f"training_curves_{cfg.model}{'_fft' if cfg.use_fft_channel else ''}.png"
    plot_training_curves(history, curves_path)
    print(f"Saved training curves → {curves_path}")

    history_path = ckpt_path.with_suffix(".history.json")
    with history_path.open("w") as f:
        json.dump(
            {
                "train_loss": history.train_loss,
                "train_acc": history.train_acc,
                "val_loss": history.val_loss,
                "val_acc": history.val_acc,
            },
            f,
            indent=2,
        )
    print(f"Saved history → {history_path}")
    print(f"Checkpoint   → {ckpt_path}")


if __name__ == "__main__":
    main()
