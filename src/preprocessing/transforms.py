"""Transform pipelines for training and evaluation.

Design choice: we keep transforms deliberately simple. The CIFAKE paper (and
recent robustness studies) show that strong augmentations — especially blur and
aggressive downscaling — actively *destroy* the spectral signal that
distinguishes AI-generated images. So we stick to mild geometric augmentations
for training and plain resize/normalize for evaluation.
"""
from __future__ import annotations

from typing import Callable

from torchvision import transforms

from src.preprocessing.frequency import FFTChannel

# ImageNet statistics — used when fine-tuning a pretrained backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CIFAR-10 statistics — closer to the REAL half of CIFAKE. Used for the
# from-scratch custom CNN so activations are well-centered from step 1.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _normalize_for(model: str) -> transforms.Normalize:
    if model == "resnet50":
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return transforms.Normalize(CIFAR_MEAN, CIFAR_STD)


def build_train_transform(
    model: str, image_size: int, use_fft_channel: bool = False
) -> Callable:
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Tiny translation only — a hard crop nukes low-freq structure
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        _normalize_for(model),
    ]
    if use_fft_channel:
        # FFT channel goes AFTER normalization so it's not rescaled by Normalize
        ops.append(FFTChannel())
    return transforms.Compose(ops)


def build_eval_transform(
    model: str, image_size: int, use_fft_channel: bool = False
) -> Callable:
    ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        _normalize_for(model),
    ]
    if use_fft_channel:
        ops.append(FFTChannel())
    return transforms.Compose(ops)
