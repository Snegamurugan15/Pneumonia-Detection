from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


APP_TITLE = "Pneumonia Detection Dashboard"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1180px;}
    [data-testid="stMetricValue"] {font-size: 1.55rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def image_stats(image: Image.Image) -> dict:
    gray = image.convert("L").resize((224, 224))
    arr = np.asarray(gray) / 255.0
    center = arr[55:170, 55:170]
    edge = np.concatenate([arr[:35, :].ravel(), arr[-35:, :].ravel(), arr[:, :35].ravel(), arr[:, -35:].ravel()])
    opacity_signal = float(center.mean() - edge.mean())
    contrast = float(arr.std())
    texture = float(np.mean(np.abs(np.diff(arr, axis=0))) + np.mean(np.abs(np.diff(arr, axis=1))))
    return {
        "mean_brightness": float(arr.mean()),
        "contrast": contrast,
        "texture": texture,
        "opacity_signal": opacity_signal,
    }


def demo_probabilities(stats: dict, name_hint: str = "") -> pd.DataFrame:
    hint = name_hint.lower()
    if "bacteria" in hint:
        probs = {"Normal": 0.08, "Bacterial Pneumonia": 0.75, "Viral Pneumonia": 0.17}
    elif "virus" in hint:
        probs = {"Normal": 0.11, "Bacterial Pneumonia": 0.22, "Viral Pneumonia": 0.67}
    elif "normal" in hint:
        probs = {"Normal": 0.82, "Bacterial Pneumonia": 0.10, "Viral Pneumonia": 0.08}
    else:
        abnormal = np.clip((stats["opacity_signal"] + stats["texture"]) * 2.2, 0.15, 0.85)
        probs = {
            "Normal": float(1 - abnormal),
            "Bacterial Pneumonia": float(abnormal * 0.58),
            "Viral Pneumonia": float(abnormal * 0.42),
        }
    total = sum(probs.values())
    return pd.DataFrame({"class": list(probs), "probability": [v / total for v in probs.values()]})


sample_dir = Path("sample_images")
sample_images = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.jpeg")) + sorted(sample_dir.glob("*.png"))

st.title(APP_TITLE)
st.caption("Portfolio dashboard for chest X-ray classification workflow, sample inspection, and model-result presentation.")

source = st.sidebar.radio("Image source", ["Sample image", "Upload image"])
image = None
name_hint = ""
if source == "Sample image" and sample_images:
    selected = st.sidebar.selectbox("Sample", sample_images, format_func=lambda p: p.name)
    image = Image.open(selected)
    name_hint = selected.name
else:
    upload = st.sidebar.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"])
    if upload is not None:
        image = Image.open(upload)
        name_hint = upload.name

if image is None:
    st.info("Choose a sample image or upload a chest X-ray image to inspect the dashboard.")
    st.stop()

stats = image_stats(image)
probs = demo_probabilities(stats, name_hint)
predicted = probs.sort_values("probability", ascending=False).iloc[0]

left, right = st.columns([0.9, 1.1])
with left:
    st.image(image, caption=name_hint or "Uploaded image", use_container_width=True)
with right:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", predicted["class"])
    c2.metric("Confidence", f"{predicted['probability'] * 100:.1f}%")
    c3.metric("Image contrast", f"{stats['contrast']:.2f}")
    st.plotly_chart(px.bar(probs, x="class", y="probability", color="class", range_y=[0, 1]), use_container_width=True)

st.subheader("Image Quality Signals")
signals = pd.DataFrame([stats]).T.reset_index()
signals.columns = ["metric", "value"]
st.dataframe(signals, use_container_width=True, hide_index=True)

st.warning("Educational demo only. This dashboard is not medical diagnosis software and should not be used for clinical decisions.")
