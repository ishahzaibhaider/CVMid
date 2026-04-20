"""Centralised configuration for VeriVision.

All hyperparameters and paths live here so the rest of the code stays clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "CIFAKE"
CKPT_DIR = PROJECT_ROOT / "models_ckpt"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CLASS_NAMES: Tuple[str, str] = ("FAKE", "REAL")  # matches ImageFolder alphabetical order


@dataclass
class TrainConfig:
    model: str = "custom_cnn"              # {"custom_cnn", "resnet50"}
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    use_fft_channel: bool = False          # adds FFT magnitude as 4th channel
    image_size: int = 32                   # 32 for custom CNN, 224 for resnet50
    val_split: float = 0.1                 # carve validation from training set
    early_stop_patience: int = 3
    checkpoint_name: str = "best.pt"
    device: str = "auto"                   # "auto" | "cuda" | "mps" | "cpu"

    def resolved_image_size(self) -> int:
        # Force 224 for resnet50 regardless of user input
        if self.model == "resnet50":
            return 224
        return self.image_size


@dataclass
class EvalConfig:
    model: str = "custom_cnn"
    checkpoint_path: str = ""
    batch_size: int = 256
    num_workers: int = 2
    use_fft_channel: bool = False
    image_size: int = 32
    device: str = "auto"
    save_figures: bool = True
