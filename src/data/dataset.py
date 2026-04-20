"""CIFAKE dataset loading.

CIFAKE ships as:
    data/CIFAKE/train/REAL/*.jpg
    data/CIFAKE/train/FAKE/*.jpg
    data/CIFAKE/test/REAL/*.jpg
    data/CIFAKE/test/FAKE/*.jpg

torchvision's ImageFolder assigns labels alphabetically:
    FAKE -> 0, REAL -> 1
which matches src.config.CLASS_NAMES.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets


class CIFAKEDataset(Dataset):
    """Thin wrapper around torchvision.ImageFolder.

    Kept as a separate class so the preprocessing module can plug in a transform
    that returns a 4-channel tensor (RGB + FFT magnitude) without surprising
    the default ImageFolder contract.
    """

    def __init__(self, root: Path, split: str, transform: Optional[Callable] = None):
        split_dir = Path(root) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Expected {split_dir} to exist. See data/README.md for download steps."
            )
        self.inner = datasets.ImageFolder(str(split_dir), transform=None)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.inner.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_dataloaders(
    root: Path,
    train_transform: Callable,
    eval_transform: Callable,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader).

    val is carved from the original train split with a fixed seed so the CIFAKE
    test set stays untouched until the final evaluation.
    """
    full_train = CIFAKEDataset(root, "train", transform=train_transform)
    test_set = CIFAKEDataset(root, "test", transform=eval_transform)

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_indices_subset = random_split(
        full_train, [n_train, n_val], generator=generator
    )

    # The val split shouldn't use the training augmentations — rebuild with eval transform
    full_train_eval = CIFAKEDataset(root, "train", transform=eval_transform)
    val_subset = Subset(full_train_eval, val_indices_subset.indices)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader, test_loader
