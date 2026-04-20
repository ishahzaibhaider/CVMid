"""Interactive Streamlit demo for VeriVision.

Run from project root:
    streamlit run app/streamlit_app.py

Features:
    - Upload an image, get REAL vs FAKE prediction + confidence
    - Toggle Grad-CAM heatmap
    - Toggle FFT spectrum visualization (standalone, for teaching)
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

uploaded = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png", "webp"])

if uploaded is None:
    st.info("Upload an image to see the prediction, or use one of the samples from the CIFAKE test set.")
    st.stop()

pil = Image.open(uploaded).convert("RGB")
tensor = transform(pil).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

pred_idx = int(np.argmax(probs))
label = CLASS_NAMES[pred_idx]
conf = float(probs[pred_idx])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    st.image(pil, use_container_width=True)

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
    disp = np.asarray(pil.resize((224, 224))).astype(np.float32) / 255.0
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
    spectrum = compute_fft_magnitude(np.asarray(pil))
    st.image(spectrum, use_container_width=True, clamp=True)
    st.caption(
        "Diffusion models tend to leave characteristic ring/cross patterns in the Fourier spectrum. "
        "Natural images usually show a smoother 1/f falloff."
    )
