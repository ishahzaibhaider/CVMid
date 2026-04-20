"""Evaluation metrics and plotting.

Everything writes to `reports/figures/` so the students can drop the PNGs
straight into a report or slide deck.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CLASS_NAMES


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(class == REAL)
        preds = logits.argmax(dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    probs, preds, labels = collect_predictions(model, loader, device)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(
            labels, preds, target_names=list(CLASS_NAMES), digits=4
        ),
        "probs": probs,
        "preds": preds,
        "labels": labels,
    }
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(CLASS_NAMES),
        yticklabels=list(CLASS_NAMES),
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc(labels: np.ndarray, probs: np.ndarray, save_path: Path) -> float:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return auc


def plot_training_curves(history, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history.train_loss, label="train")
    axes[0].plot(epochs, history.val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history.train_acc, label="train")
    axes[1].plot(epochs, history.val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
