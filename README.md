# VeriVision — Real vs AI-Generated Image Detector

> **Introduction to Computer Vision — Mid Project**
> Detecting AI-generated images using deep learning + frequency-domain analysis.

A fully working pipeline that (a) takes the CIFAKE dataset, (b) applies a unique preprocessing stack combining standard transforms with FFT-based spectral features, (c) trains two models (a from-scratch CNN and a ResNet-50 transfer-learning model), and (d) produces predictions with Grad-CAM visualisations that show *why* the model called an image real or fake.

---

## Team

| Name | Registration |
|---|---|
| **Shahzaib Haider Rizvi** | FA23-BAI-050 |
| Aymen Ali Seemab | FA23-BAI-011 |
| Nimra Tahseen Yousaf | FA23-BAI-043 |

---

## Problem statement

Generative models (Stable Diffusion, Midjourney, DALL·E, etc.) now produce photorealistic images that are indistinguishable from real photographs to the human eye. This has surfaced very real harms in 2025–2026 — fake news imagery, social-media deepfakes, AI-generated art passing as human-made, and identity fraud. A reliable, interpretable classifier that flags AI-generated images is therefore an actively useful tool for journalists, content moderators, and everyday users.

**Our task:** binary classification — given an image, predict whether it was captured by a camera (`REAL`) or synthesised by a diffusion model (`FAKE`). We also want the system to *explain* itself via Grad-CAM heatmaps, so we can see the visual evidence that drove each prediction.

---

## Why this problem is a good fit for the mid project

| Instruction requirement | How we meet it |
|---|---|
| Align to a real problem statement | AI-image detection — socially relevant, 2026-topical |
| Relevant pre-processing | Standard (resize, normalize, flip, translate) **plus** FFT magnitude as a 4th channel — diffusion models leave spectral fingerprints |
| Train an ML / DL model | Custom CNN (from scratch) **and** ResNet-50 (transfer learning), side-by-side comparison |
| Show predictions | CLI predictor, Streamlit demo app, Grad-CAM overlays, full metrics report |

---

## What's unique about this project

1. **FFT-channel preprocessing.** Most student projects stop at `resize + normalize`. We add a frequency-domain channel (log-magnitude of the 2D FFT) alongside RGB. Published work shows diffusion models leave characteristic high-frequency artefacts; we make those artefacts available to the model as an explicit input feature.
2. **Two-model comparison.** A shallow custom CNN establishes the from-scratch baseline; a ResNet-50 transfer-learning model shows the ceiling. Target accuracies: ~92 % and ~97 % respectively on the CIFAKE test set.
3. **Grad-CAM explainability.** Predictions come with heatmaps. The Bird & Lotfi CIFAKE paper (IEEE Access 2024) found detectors focus on *background texture* rather than foreground objects — our Grad-CAM grids reproduce and visualise this finding.
4. **Live Streamlit demo.** Upload an image, get a verdict + confidence + heatmap + FFT spectrum in real time.

---

## Repository layout

```
CVMid/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md                    # How to download CIFAKE
├── src/
│   ├── config.py                    # All hyperparameters and paths
│   ├── utils.py                     # Seed, device, checkpoint helpers
│   ├── data/dataset.py              # ImageFolder wrapper + DataLoaders
│   ├── preprocessing/
│   │   ├── transforms.py            # Train / eval transform pipelines
│   │   └── frequency.py             # FFT channel + spectrum helpers
│   ├── models/
│   │   ├── custom_cnn.py            # Baseline CNN (≈0.2M params)
│   │   ├── transfer_resnet.py       # ResNet-50 transfer-learning head
│   │   └── factory.py               # build_model() dispatcher
│   ├── training/trainer.py          # Training loop + early stopping
│   └── evaluation/
│       ├── metrics.py               # Accuracy / precision / recall / F1 / ROC
│       └── gradcam.py               # Grad-CAM with a safe fallback
├── scripts/
│   ├── train.py                     # Train a model
│   ├── evaluate.py                  # Evaluate on test set + plots
│   ├── predict.py                   # Predict on single image or folder
│   └── demo.py                      # Regenerate all report figures
├── app/streamlit_app.py             # Interactive web demo
├── models_ckpt/                     # Checkpoints land here (gitignored)
├── reports/
│   ├── figures/                     # Plots / heatmaps (gitignored)
│   └── metrics_*.json               # Test metrics per run
└── docs/
    ├── project_proposal.md
    ├── methodology.md
    └── presentation_outline.md
```

---

## Setup

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd CVMid

# 2. Create a virtual env (recommended)
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate            # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the CIFAKE dataset — see data/README.md
#    End result must be: data/CIFAKE/{train,test}/{REAL,FAKE}/*.jpg
```

---

## How to run

All commands are run from the project root.

### Train the baseline CNN (fastest — ~10 min on CPU, ~2 min on GPU)

```bash
python -m scripts.train --model custom_cnn --epochs 10
```

### Train the ResNet-50 transfer-learning model

```bash
python -m scripts.train --model resnet50 --epochs 5 --batch-size 64
```

### Train the FFT-channel variant (our unique preprocessing)

```bash
python -m scripts.train --model custom_cnn --use-fft-channel --epochs 10
```

### Evaluate on the held-out CIFAKE test set

```bash
python -m scripts.evaluate --checkpoint models_ckpt/custom_cnn.pt
```

Produces accuracy / precision / recall / F1 / ROC-AUC, a confusion matrix PNG, and an ROC curve PNG under `reports/figures/`.

### Predict on your own images

```bash
# Single image
python -m scripts.predict --checkpoint models_ckpt/custom_cnn.pt --image path/to/image.jpg

# Whole folder, with Grad-CAM overlays saved
python -m scripts.predict --checkpoint models_ckpt/custom_cnn.pt --dir my_images/ --gradcam
```

### Launch the Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Drag-drop an image, see the verdict, Grad-CAM heatmap, and FFT spectrum side by side.

### Train on free GPU + deploy the demo online

See [**docs/deployment.md**](docs/deployment.md) for the full walkthrough:
train on **Kaggle** (free T4 GPU, CIFAKE pre-hosted) and deploy on
**Hugging Face Spaces** (free Streamlit hosting). Total cost: ₨ 0.

### Regenerate all figures for the report / slides

```bash
python -m scripts.demo
```

---

## Expected results

Based on published CIFAKE benchmarks and our architecture choices:

| Model | Input | Target accuracy | Target ROC-AUC |
|---|---|---|---|
| Custom CNN (from scratch) | 32×32 RGB | ~92 % | ~0.98 |
| Custom CNN + FFT channel | 32×32 RGB + FFT | ~93–94 % | ~0.98 |
| ResNet-50 (transfer learning) | 224×224 RGB | ~97 % | ~0.997 |

See `docs/methodology.md` for the full experimental setup and `reports/metrics_*.json` for actual numbers after you train.

---

## Notes on reproducibility

- Fixed seed (`--seed 42`) across `random`, `numpy`, and `torch`.
- `cudnn.deterministic = True` (small throughput cost, bit-exact runs).
- Validation split is carved from the original train split with a stable seed; the CIFAKE **test** split is never touched during training.

---

## References

1. Bird, J. J., & Lotfi, A. (2024). **CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.** IEEE Access.
2. Selvaraju, R. R., et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.** ICCV.
3. He, K., et al. (2016). **Deep Residual Learning for Image Recognition.** CVPR.
4. *Methods and Trends in Detecting Generated Images: A Survey*, arXiv:2502.15176 (2025).
5. *Universal detection of synthetic images using frequency-domain fingerprints (UGAD)*, arXiv:2409.07913 (2024).

---

*Built for the Introduction to Computer Vision course, mid-project submission.*
