
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import torch

# Patch torch.classes.__path__ to suppress Streamlit module inspection warning
if hasattr(torch, "classes"):
    try:
        torch.classes.__path__ = []
    except Exception:
        pass
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path

from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
from scene_motion_llm.inference import SceneMotionInferencer

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SceneMotion AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# MODERN GLASS UI
# =========================================================
st.markdown("""
<style>

.stApp { background: #f7f8fc; }
.block-container { max-width: 1080px; padding-top: 3rem; padding-bottom: 3rem; }

/* Glass card */
.glass { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.4rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); }

/* Title */
.title { font-size: 2.5rem; font-weight: 700; color: #172554; }

/* Subtitle */
.subtitle { color: #64748b; margin-bottom: 1.5rem; }

/* Sentiment styles */
.pos { color: #15803d; font-size: 1.75rem; font-weight: 700; }
.neu { color: #a16207; font-size: 1.75rem; font-weight: 700; }
.neg { color: #b91c1c; font-size: 1.75rem; font-weight: 700; }
div.stButton > button { background: #1d4ed8; color: white; border: 0; border-radius: 9px; font-weight: 600; }
div.stButton > button:hover { background: #1e40af; color: white; }

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">🎬 SceneMotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a video and get a concise scene-motion sentiment prediction.</div>', unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "model" not in st.session_state:
    st.session_state.model = None

if "inferencer" not in st.session_state:
    st.session_state.inferencer = None

if "device" not in st.session_state:
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD MODEL
# =========================================================
def load_model():
    if st.session_state.model is None:

        with st.spinner("Loading AI Model..."):

            model = SceneMotionLLMModel(
                spatial_feature_dim=512,
                motion_feature_dim=2,
                temporal_hidden_dim=256,
                attention_dim=128,
                fusion_dim=512,
                num_classes=3,
                max_frames=30,
                dropout=0.3
            )

            checkpoint_path = (
                Path(__file__).resolve().parent
                / "checkpoints"
                / "stage2_accede"
                / "best_model.pt"
            )

            if not checkpoint_path.exists():
                st.error("Compatible LIRIS sentiment checkpoint was not found.")
                st.stop()

            inferencer = SceneMotionInferencer(
                model=model,
                checkpoint_path=str(checkpoint_path),
                device=st.session_state.device,
                max_frames=30
            )
            st.session_state.model = inferencer.model
            st.session_state.inferencer = inferencer
            st.success("✅ LIRIS sentiment model loaded")

# =========================================================
# SAFE RESULT HANDLER
# =========================================================
def safe_get(result, key, default=None):
    return result.get(key, default)

# =========================================================
# DISPLAY RESULT
# =========================================================
def show_result(result):

    sentiment = result.get("sentiment", "Unknown")
    confidence = result.get("confidence", 0.0)
    probs = result.get("probabilities", [0, 0, 0])

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🎯 Prediction Result")

    if sentiment == "Positive":
        st.markdown(f'<div class="pos">{sentiment}</div>', unsafe_allow_html=True)
    elif sentiment == "Neutral":
        st.markdown(f'<div class="neu">{sentiment}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="neg">{sentiment}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{confidence:.2%}")
    col2.metric("Device", st.session_state.device.upper())

    st.subheader("Class probabilities")
    labels = ["Positive", "Neutral", "Negative"]

    for i in range(3):
        st.write(f"{labels[i]}")
        st.progress(float(probs[i]))
        st.write(f"{probs[i]:.2%}")

    model_evidence = result.get("model_evidence", {})
    evidence = result.get("evidence", {})
    if model_evidence or evidence:
        st.divider()
        st.subheader("Research evidence")
        evidence_columns = st.columns(4)
        evidence_columns[0].metric("Frames analyzed", result.get("num_frames", 0))
        evidence_columns[1].metric("Probability margin", f"{model_evidence.get('probability_margin', 0.0):.2%}")
        evidence_columns[2].metric("Mean motion", f"{evidence.get('mean_motion', 0.0):.3f}")
        evidence_columns[3].metric("Frame change", f"{evidence.get('frame_change', 0.0):.3f}")

        st.caption(
            "Top-attended frames are the sampled frames given the most temporal attention by the model. "
            "They are inspection evidence, not a causal explanation."
        )
        top_frames = model_evidence.get("top_attended_frames", [])
        if top_frames:
            st.dataframe(top_frames, hide_index=True, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# VIDEO ANALYSIS
# =========================================================
def analyze_video(video_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        path = tmp.name

    try:
        load_model()

        with st.spinner("AI is analyzing video..."):

            result = st.session_state.inferencer.analyze_video(path)

        st.success("Analysis Complete")

        show_result(result)

        # =========================
        # SAFE OPTIONAL SECTIONS
        # =========================

        with st.expander("Technical report", expanded=False):
            st.text(result.get("report", "No report generated."))

        # DOWNLOAD
        download_data = {
            "sentiment": result.get("sentiment"),
            "confidence": float(result.get("confidence", 0)),
            "probabilities": result.get("probabilities", []).tolist() if hasattr(result.get("probabilities"), "tolist") else result.get("probabilities"),
            "visual_evidence": result.get("evidence", {}),
            "model_evidence": result.get("model_evidence", {}),
            "timestamp": datetime.now().isoformat()
        }

        st.download_button(
            "⬇ Download JSON",
            json.dumps(download_data, indent=2),
            "result.json",
            "application/json"
        )

    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

# =========================================================
# UI SECTION
# =========================================================
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Upload a video")
    file = st.file_uploader("Supported: MP4, AVI, MOV, MKV", type=["mp4", "avi", "mov", "mkv"])

    if file:
        st.caption(f"Ready to analyze: {file.name} · {file.size / 1024 / 1024:.1f} MB")
        st.video(file)
        if st.button("Analyze video", type="primary", use_container_width=True):
            analyze_video(file)
    else:
        st.info("Choose a video to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("How it works")
    st.markdown("""
    1. Upload a short video.
    2. SceneMotion samples frames and motion.
    3. Review the predicted sentiment and confidence.
    """)
    st.divider()
    st.caption(f"Running on {st.session_state.device.upper()} · Results stay on this device")
    st.markdown("</div>", unsafe_allow_html=True)
