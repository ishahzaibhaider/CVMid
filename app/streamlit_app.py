"""Interactive Streamlit demo for VeriVision.

Run from project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CKPT_DIR, CLASS_NAMES  # noqa: E402
from src.evaluation.gradcam import gradcam_overlay  # noqa: E402
from src.models import build_model, gradcam_target_layer  # noqa: E402
from src.preprocessing import build_eval_transform  # noqa: E402
from src.preprocessing.frequency import compute_fft_magnitude  # noqa: E402
from src.utils import resolve_device  # noqa: E402


st.set_page_config(page_title="VeriVision — Real vs AI-Generated Image Detector", layout="wide")

st.title("VeriVision")
st.caption("Detect AI-generated images using CNN + frequency-domain analysis  ·  Computer Vision Mid Project")

SAMPLES_DIR = ROOT / "app" / "samples"


with st.sidebar:
    st.header("Team")
    st.markdown(
        "- **Shahzaib Haider Rizvi** · FA23-BAI-050\n"
        "- **Aymen Ali Seemab** · FA23-BAI-011\n"
        "- **Nimra Tahseen Yousaf** · FA23-BAI-043"
    )
    st.divider()
    st.header("Model")
    available = sorted(CKPT_DIR.glob("*.pt"))
    if not available:
        st.error("No checkpoints found in `models_ckpt/`. Train a model first (see README).")
        st.stop()
    ckpt_path = st.selectbox(
        "Checkpoint",
        options=available,
        format_func=lambda p: p.name,
    )
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    show_fft = st.checkbox("Show FFT spectrum", value=True)

    st.divider()
    with st.expander("ℹ️ About this model — read before testing"):
        st.markdown(
            "This model was trained on the **CIFAKE** dataset:\n"
            "- REAL images from CIFAR-10 (natural photos)\n"
            "- FAKE images from **Stable Diffusion v1.4** (2022)\n"
            "- All at **32 × 32 resolution**\n\n"
            "**Test-set accuracy (in-distribution):**\n"
            "- Custom CNN: **96.77 %**\n"
            "- ResNet-50:  **98.26 %**\n\n"
            "**Known limitations (out-of-distribution):**\n"
            "- Newer generators (Midjourney, DALL·E 3, **Nano Banana**, SDXL) leave different fingerprints — we never trained on those, so accuracy drops.\n"
            "- High-resolution images get downsampled to 32×32 for the model, which destroys most of the forensic signal.\n\n"
            "→ For a representative demo, use the 'Try a sample' section below."
        )

    st.divider()


# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def load_model(ckpt: Path):
    device = resolve_device("auto")
    blob = torch.load(ckpt, map_location=device)
    meta = blob.get("meta", {})
    model_name = meta.get("model", "custom_cnn")
    in_channels = meta.get("in_channels", 3)
    use_fft = meta.get("use_fft_channel", False)
    image_size = meta.get("image_size", 32 if model_name == "custom_cnn" else 224)
    model, _ = build_model(model_name, in_channels=in_channels, num_classes=2)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    target_layer = gradcam_target_layer(model, model_name)
    tf = build_eval_transform(model_name, image_size, use_fft)
    return model, target_layer, tf, device, meta


model, target_layer, transform, device, meta = load_model(ckpt_path)
st.sidebar.success(
    f"Loaded **{meta.get('model')}** · image_size={meta.get('image_size')} · fft={meta.get('use_fft_channel')}"
)


# ── Image source selection ───────────────────────────────────────────────────
st.subheader("Pick an image to test")

sample_files = sorted(SAMPLES_DIR.glob("*.jpg")) + sorted(SAMPLES_DIR.glob("*.png"))
tabs = st.tabs(["📁 Try a sample (recommended)", "⬆️ Upload your own"])

selected_image: Image.Image | None = None
warning: str | None = None

with tabs[0]:
    if not sample_files:
        st.info(
            "No sample images yet. From the project root, run:\n\n"
            "`python -m scripts.fetch_samples`\n\n"
            "This downloads ~10 CIFAKE test images (5 REAL + 5 FAKE)."
        )
    else:
        cols = st.columns(min(len(sample_files), 5))
        for i, p in enumerate(sample_files):
            with cols[i % len(cols)]:
                st.image(str(p), caption=p.stem, use_container_width=True)
        choice = st.selectbox("Load sample", options=sample_files, format_func=lambda p: p.name)
        if st.button("Use this sample"):
            st.session_state["selected_path"] = str(choice)

with tabs[1]:
    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader",
    )
    st.caption(
        "⚠️ Note: this model was trained on 32×32 SD-v1.4 images. "
        "Very large or non-SD-v1.4 images may be misclassified — this is a known limitation, not a bug."
    )
    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
        selected_image = pil
        w, h = pil.size
        if max(w, h) > 256:
            warning = (
                f"Uploaded image is {w}×{h}. It will be downsampled to "
                f"{meta.get('image_size')}×{meta.get('image_size')} for the model — "
                "most forensic signal will be lost in the resize. Expect OOD behavior."
            )

if "selected_path" in st.session_state and selected_image is None:
    p = Path(st.session_state["selected_path"])
    if p.exists():
        selected_image = Image.open(p).convert("RGB")

if selected_image is None:
    st.stop()


# ── Predict ──────────────────────────────────────────────────────────────────
tensor = transform(selected_image).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

pred_idx = int(np.argmax(probs))
label = CLASS_NAMES[pred_idx]
conf = float(probs[pred_idx])

if warning:
    st.warning(warning)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    st.image(selected_image, use_container_width=True)

with col2:
    st.subheader("Prediction")
    if label == "FAKE":
        st.error(f"⚠️  {label}  ·  {conf:.1%} confidence")
    else:
        st.success(f"✅  {label}  ·  {conf:.1%} confidence")
    st.progress(float(probs[0]), text=f"P[FAKE] = {probs[0]:.3f}")
    st.progress(float(probs[1]), text=f"P[REAL] = {probs[1]:.3f}")

if show_gradcam:
    st.subheader("Grad-CAM — where the model is looking")
    disp = np.asarray(selected_image.resize((224, 224))).astype(np.float32) / 255.0
    overlay = gradcam_overlay(
        model=model,
        target_layer=target_layer,
        input_tensor=tensor,
        original_rgb=disp,
        target_class=pred_idx,
    )
    st.image(overlay, use_container_width=True, clamp=True)
    st.caption(
        "Red = regions that pushed the prediction toward **" + label + "**. "
        "Prior work shows AI-detectors often focus on background texture and high-freq artifacts."
    )

if show_fft:
    st.subheader("FFT magnitude spectrum")
    spectrum = compute_fft_magnitude(np.asarray(selected_image))
    st.image(spectrum, use_container_width=True, clamp=True)
    st.caption(
        "Diffusion models tend to leave characteristic ring/cross patterns in the Fourier spectrum. "
        "Natural images usually show a smoother 1/f falloff."
    )
