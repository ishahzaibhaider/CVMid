"""Training loop with early stopping, checkpointing, and history logging."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float,
        weight_decay: float,
        epochs: int,
        early_stop_patience: int,
        checkpoint_path: Path,
        meta: Dict[str, Any],
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = early_stop_patience
        self.checkpoint_path = checkpoint_path
        self.meta = meta

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.history = TrainHistory()

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        bar = tqdm(loader, desc="train" if train else "val", leave=False)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for images, labels in bar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                loss_sum += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)
                bar.set_postfix(loss=loss_sum / total, acc=correct / total)
        return {"loss": loss_sum / total, "acc": correct / total}

    def fit(self) -> TrainHistory:
        best_val = float("inf")
        epochs_without_improve = 0
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            tr = self._run_epoch(self.train_loader, train=True)
            va = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            self.history.train_loss.append(tr["loss"])
            self.history.train_acc.append(tr["acc"])
            self.history.val_loss.append(va["loss"])
            self.history.val_acc.append(va["acc"])

            print(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                f"val loss {va['loss']:.4f} acc {va['acc']:.4f}"
            )

            if va["loss"] < best_val - 1e-4:
                best_val = va["loss"]
                epochs_without_improve = 0
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "epoch": epoch,
                        "val_loss": va["loss"],
                        "val_acc": va["acc"],
                        "meta": self.meta,
                    },
                    self.checkpoint_path,
                )
                print(f"  ↳ new best (val_loss={va['loss']:.4f}) — checkpoint saved")
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= self.patience:
                    print(f"Early stopping after {epoch} epochs (no improvement for {self.patience}).")
                    break

        return self.history
