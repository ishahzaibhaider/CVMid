# Methodology — VeriVision

This document walks through every design decision in the pipeline and the reasoning behind it. It mirrors what should go into the methodology section of the written report.

---

## 1. Dataset

**CIFAKE** ships as four flat ImageFolder directories: `train/{REAL,FAKE}` and `test/{REAL,FAKE}`, 50 k + 50 k + 10 k + 10 k respectively. All images are 32 × 32 RGB.

- REAL half: CIFAR-10 (10 natural-image classes: airplane, bird, cat, deer, dog, frog, horse, ship, truck, automobile).
- FAKE half: Stable Diffusion v1.4, prompted to match each CIFAR-10 class.

**Train / validation split.** We carve a 10 % validation set out of the 100 k training images with a fixed seed (42). The CIFAKE test set (20 k images) is held out completely and only touched by `scripts/evaluate.py` at the very end — no hyperparameter tuning against it.

**Label convention.** `torchvision.datasets.ImageFolder` assigns class indices alphabetically: `FAKE → 0`, `REAL → 1`. Confusion matrices, classification reports, and ROC curves all follow this convention.

---

## 2. Pre-processing

### 2.1 Standard transforms

| Step | Train | Validation / Test |
|---|---|---|
| Resize | → target size | → target size |
| Random horizontal flip | p = 0.5 | — |
| Random affine translation | ±5 % | — |
| ToTensor | ✓ | ✓ |
| Normalise | ✓ | ✓ |

Target size is **32 × 32** for the custom CNN (native CIFAKE resolution) and **224 × 224** for ResNet-50 (ImageNet pretraining expects 224).

Normalisation statistics:
- Custom CNN → CIFAR-10 statistics `mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)` — closer to the REAL half of CIFAKE, so activations start well-centred.
- ResNet-50 → ImageNet statistics `mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)` — matches the distribution the pretrained weights expect.

### 2.2 What we deliberately *don't* do

Recent (2025) robustness studies on CIFAKE show that heavy Gaussian blur drops detector accuracy from ~93 % to ~26 %, and aggressive downscaling drops it to ~32 %. Blur in particular destroys exactly the high-frequency structure that separates REAL from FAKE. So:

- No Gaussian blur augmentation
- No random resized crop to a smaller size
- No colour jitter beyond normalisation
- No JPEG re-compression

This is a conscious, defensible choice, not an oversight.

### 2.3 FFT channel (our contribution)

Diffusion models leave characteristic artefacts in the frequency domain — cross / ring patterns in the magnitude spectrum, elevated high-frequency energy, and non-natural 1/f falloff shape (see UGAD, DEFEND). Rather than train a separate spectral-only model, we make the spectrum available to the same CNN as a **fourth input channel**.

Procedure (see `src/preprocessing/frequency.py`):

1. Convert RGB → luminance (`Y = 0.299 R + 0.587 G + 0.114 B`).
2. Compute `F = fftshift(fft2(Y))`.
3. Take `log(1 + |F|)` — log compresses the dynamic range so the DC component doesn't dominate.
4. Min-max normalise to [0, 1].
5. Concatenate as channel 3 of the input tensor, producing a (4, H, W) tensor.

When ResNet-50 is used with the FFT channel, we surgically replace its first conv from 3 → 4 input channels. The RGB filter weights are copied as-is; the extra channel's weights are initialised as the mean across the original RGB filter weights. This preserves the ImageNet features almost perfectly while letting the model learn to use the new channel.

---

## 3. Models

### 3.1 Custom CNN

Three conv blocks, each a doubled Conv-BN-ReLU followed by a 2 × 2 max-pool. Global average pooling then a two-layer MLP head with dropout. ~200 k trainable parameters. Operates at native 32 × 32 resolution — upsampling a 32 × 32 image to 224 provides no new information and just wastes compute.

```
Input (3 or 4, 32, 32)
  → ConvBlock(→32)  [32×32 → 16×16]
  → ConvBlock(→64)  [16×16 → 8×8]
  → ConvBlock(→128) [8×8   → 4×4]
  → GAP             [128×1×1]
  → Dropout(0.3) → Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 2)
```

### 3.2 ResNet-50 (transfer learning)

Standard torchvision `resnet50` with `IMAGENET1K_V2` pretrained weights. Final `fc` layer replaced with `Dropout(0.3) → Linear(2048, 2)`. All layers fine-tuned (not frozen) because CIFAKE is plenty big enough to afford full fine-tuning and the pretrained features are only approximately aligned with the REAL-vs-FAKE task.

---

## 4. Training

- **Optimiser.** AdamW (lr 1e-3, weight decay 1e-4) — AdamW is the standard choice for both from-scratch and fine-tuning regimes.
- **Scheduler.** Cosine annealing over the full epoch budget. Removes the need to tune a step-schedule by hand.
- **Loss.** Cross-entropy.
- **Early stopping.** Patience 3 on validation loss. Best-checkpoint kept, not the last one.
- **Batch size.** 128 for the custom CNN, 64 for ResNet-50 (larger feature maps eat more memory).
- **Epoch budget.** 10 for custom CNN, 5 for ResNet-50 (pretrained model converges faster).
- **Reproducibility.** Seeded `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`; `cudnn.deterministic = True`.

---

## 5. Evaluation

- **Accuracy, precision, recall, F1** — per class and macro-averaged (sklearn's `classification_report`).
- **ROC-AUC** on `P(class = REAL)` probabilities.
- **Confusion matrix** saved as a heatmap PNG.
- **ROC curve** saved as a PNG.
- **Training curves** (loss and accuracy, train vs val) saved as a PNG.
- **Metrics JSON** dropped into `reports/metrics_<model>.json` for easy inclusion in tables.

---

## 6. Explainability (Grad-CAM)

`src/evaluation/gradcam.py` wraps `pytorch-grad-cam` when it's installed, with a hand-rolled fallback. Target layer:

- Custom CNN → the final `conv2` inside the last `ConvBlock` (4 × 4 feature map).
- ResNet-50 → `layer4[-1]` (the final bottleneck block, 7 × 7 feature map).

Heatmaps are upsampled to 224 × 224 via bilinear interpolation and blended with the original image at 40 % opacity. Displaying the input at 224 × 224 — even when the model saw 32 × 32 — makes the heatmaps readable.

**What we expect to see.** The Bird & Lotfi paper reports that CIFAKE detectors focus on *background textures* and not on foreground objects. Our Grad-CAM grid is designed to reproduce that finding visually.

---

## 7. Limitations and honest caveats

- **Resolution.** CIFAKE is 32 × 32. Real-world AI-generated content is typically 512 × 512 or higher; our classifier may not transfer directly.
- **Generator coverage.** FAKE images in CIFAKE come from Stable Diffusion v1.4 only. A classifier trained on these may under-generalise to Midjourney, DALL·E 3, or newer SD versions.
- **Semantic coverage.** Only CIFAR-10 classes. No faces, no text, no complex scenes.
- **Adversarial robustness.** We explicitly don't evaluate against adversarial perturbations — that's outside the scope of a mid-project but is a reasonable extension.

These limitations are what make this a *mid-term* project rather than a finished product. They're the natural starting point for a follow-up final-project extension.
