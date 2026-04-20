# Presentation Outline — VeriVision

A suggested 10-slide deck for the mid-project presentation. Aim for ~8 minutes + 2 minutes of Q&A.

---

## Slide 1 — Title

- **VeriVision: Real vs AI-Generated Image Detector**
- Course: Introduction to Computer Vision — Mid Project
- Team: Shahzaib Haider Rizvi (FA23-BAI-050), Aymen Ali Seemab (FA23-BAI-011), Nimra Tahseen Yousaf (FA23-BAI-043)

## Slide 2 — The problem

- Diffusion models now produce images humans can't distinguish from real photos.
- Real-world harms: fake news imagery, identity fraud, AI art passing as human-made.
- Need: fast, interpretable detection.
- *Optional:* show two side-by-side images — "one of these was shot on a camera, one was made by Stable Diffusion" — audience votes.

## Slide 3 — Dataset: CIFAKE

- 120,000 images, 32 × 32 RGB, perfectly balanced.
- REAL half = CIFAR-10, FAKE half = Stable Diffusion v1.4.
- 100 k train / 20 k test, class-balanced.
- Insert `reports/figures/sample_grid.png`.

## Slide 4 — Pre-processing (standard)

- Resize, horizontal flip, mild translation, per-channel normalisation.
- What we *don't* do: blur, heavy downscale, colour jitter — they destroy the spectral signal.
- One-line justification with a reference to the 2025 robustness study.

## Slide 5 — Pre-processing (unique: FFT channel)

- Diffusion models leave fingerprints in the Fourier domain.
- We append `log(1 + |FFT(Y)|)` as a 4th input channel.
- Insert `reports/figures/fft_comparison.png` — the difference spectrum makes the fingerprint visible.
- One-liner: *"We give the model the spectrum directly instead of hoping a 3×3 conv rediscovers it."*

## Slide 6 — Models

- Two models, head-to-head:
  - Custom CNN — 3 conv blocks, GAP, MLP head, ~200 k params, native 32 × 32.
  - ResNet-50 — ImageNet-pretrained, upsampled to 224 × 224, fine-tuned end-to-end.
- Architecture diagram (or a simple ASCII block).

## Slide 7 — Training

- AdamW + cosine annealing, cross-entropy, early stopping patience 3.
- Fixed seed, deterministic cuDNN.
- Insert `reports/figures/training_curves_custom_cnn.png`.

## Slide 8 — Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Custom CNN | XX.X % | 0.XXX | 0.XXX |
| Custom CNN + FFT | XX.X % | 0.XXX | 0.XXX |
| ResNet-50 | XX.X % | 0.XXX | 0.XXX |

- Fill from `reports/metrics_*.json` after running `scripts/evaluate.py`.
- Insert `reports/figures/confusion_resnet50.png` and `reports/figures/roc_resnet50.png`.

## Slide 9 — Explainability

- Grad-CAM on correct and incorrect predictions.
- Insert `reports/figures/gradcam_grid.png`.
- Key observation: the model fixates on *background texture*, not the foreground object (consistent with Bird & Lotfi 2024).

## Slide 10 — Live demo + next steps

- Switch to browser → `streamlit run app/streamlit_app.py` → drag-and-drop an image from the test set → show prediction, Grad-CAM, FFT.
- Limitations: 32 × 32 resolution, single generator (SD v1.4), CIFAR-10 classes only.
- Future work: higher resolutions, multi-generator dataset, adversarial robustness.
- Thank you + Q&A.

---

## Speaker-note tips

- When explaining the FFT channel, resist the urge to go deep on Fourier theory — one sentence on "the spectrum exposes high-frequency artefacts" is plenty for this audience.
- If asked *"why not just train on 512 × 512 images?"* → honestly: CIFAKE is what we have at mid-project scale; the pipeline is resolution-agnostic and we can upgrade the dataset later.
- If asked *"does Grad-CAM prove it isn't overfitting?"* → no, but it does show the model isn't latching onto obviously wrong features (e.g., class-specific objects instead of general artefacts).
