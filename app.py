
# """
# Streamlit Web App for SceneMotion-LLM
# Interactive video sentiment analysis interface
# """

# import streamlit as st
# import torch
# import numpy as np
# import cv2
# import os
# import tempfile
# from pathlib import Path
# import json
# from datetime import datetime
# import logging

# # LOCAL IMPORTS
# from models.sentiment_classifier import SceneMotionLLMModel
# from inference import SceneMotionInferencer
# from utils.frame_extractor import FrameExtractor, normalize_frame
# from utils.optical_flow import OpticalFlowProcessor

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # =========================================================
# # PAGE CONFIG
# # =========================================================

# st.set_page_config(
#     page_title="SceneMotion-LLM",
#     page_icon="🎬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # =========================================================
# # CUSTOM CSS
# # =========================================================

# st.markdown("""
# <style>
# .main {
#     padding: 2rem;
# }

# .sentiment-positive {
#     color: #2ecc71;
#     font-weight: bold;
#     font-size: 24px;
# }

# .sentiment-neutral {
#     color: #f39c12;
#     font-weight: bold;
#     font-size: 24px;
# }

# .sentiment-negative {
#     color: #e74c3c;
#     font-weight: bold;
#     font-size: 24px;
# }
# </style>
# """, unsafe_allow_html=True)

# # =========================================================
# # TITLE
# # =========================================================

# st.title("🎬 SceneMotion-LLM")
# st.markdown("### Video Sentiment Analysis using Deep Learning")

# # =========================================================
# # SESSION STATE
# # =========================================================

# if "model" not in st.session_state:
#     st.session_state.model = None

# if "inferencer" not in st.session_state:
#     st.session_state.inferencer = None

# if "device" not in st.session_state:
#     st.session_state.device = (
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

# # =========================================================
# # LOAD MODEL
# # =========================================================

# def load_model_and_inferencer():

#     if st.session_state.model is None:

#         with st.spinner("Loading AI Model..."):

#             model = SceneMotionLLMModel(
#                 spatial_feature_dim=512,
#                 motion_feature_dim=2,
#                 temporal_hidden_dim=256,
#                 attention_dim=128,
#                 fusion_dim=512,
#                 num_classes=3,
#                 max_frames=30,
#                 dropout=0.3,
                
#             )

#             checkpoint_path = "./checkpoints/best_model.pt"

#             if os.path.exists(checkpoint_path):

#                 checkpoint = torch.load(
#                     checkpoint_path,
#                     map_location=st.session_state.device
#                 )

#                 model.load_state_dict(
#                     checkpoint["model_state_dict"]
#                 )

#                 st.success("✅ Pretrained Model Loaded")

#             else:
#                 st.warning(
#                     "⚠ No trained checkpoint found.\n"
#                     "Using random weights."
#                 )

#             model.to(st.session_state.device)
#             model.eval()

#             st.session_state.model = model

#             st.session_state.inferencer = SceneMotionInferencer(
#                 model=model,
#                 checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
#                 device=st.session_state.device,
#                 max_frames=30
#             )

# # =========================================================
# # DISPLAY RESULT
# # =========================================================

# def display_sentiment_result(sentiment, confidence, probabilities):

#     st.subheader("🎯 Prediction Result")

#     if sentiment.lower() == "positive":
#         st.markdown(
#             f'<p class="sentiment-positive">{sentiment}</p>',
#             unsafe_allow_html=True
#         )

#     elif sentiment.lower() == "neutral":
#         st.markdown(
#             f'<p class="sentiment-neutral">{sentiment}</p>',
#             unsafe_allow_html=True
#         )

#     else:
#         st.markdown(
#             f'<p class="sentiment-negative">{sentiment}</p>',
#             unsafe_allow_html=True
#         )

#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric(
#             "Confidence",
#             f"{confidence:.2%}"
#         )

#     with col2:
#         st.metric(
#             "Device",
#             st.session_state.device.upper()
#         )

#     st.subheader("📊 Probability Distribution")

#     sentiments = [
#         "Positive",
#         "Neutral",
#         "Negative"
#     ]

#     for i, s in enumerate(sentiments):

#         st.write(f"**{s}**")

#         st.progress(float(probabilities[i]))

#         st.write(f"{probabilities[i]:.2%}")

# # =========================================================
# # VIDEO ANALYSIS
# # =========================================================

# def analyze_video_file(video_file):

#     with tempfile.NamedTemporaryFile(
#         delete=False,
#         suffix=".mp4"
#     ) as tmp_file:

#         tmp_file.write(video_file.read())

#         tmp_path = tmp_file.name

#     try:

#         output_dir = tempfile.mkdtemp()

#         with st.spinner("Analyzing Video..."):

#             result = st.session_state.inferencer.analyze_video(
#                 tmp_path,
#                 output_dir=output_dir,
#                 save_frames=False
#             )

#         st.success("✅ Analysis Complete")

#         # DISPLAY RESULT
#         display_sentiment_result(
#             result["sentiment"],
#             result["confidence"],
#             result["probabilities"]
#         )

#         # REPORT
#         st.subheader("📝 Analysis Report")

#         st.text(result["report"])

#         # MOTION ANALYSIS
#         st.subheader("📈 Motion Analysis")

#         st.write(result["motion_analysis"])

#         # METRICS
#         metrics = result["motion_features"]

#         col1, col2, col3, col4 = st.columns(4)

#         with col1:
#             st.metric(
#                 "Mean Motion",
#                 f"{metrics['mean_magnitude']:.2f}"
#             )

#         with col2:
#             st.metric(
#                 "Max Motion",
#                 f"{metrics['max_magnitude']:.2f}"
#             )

#         with col3:
#             st.metric(
#                 "Std Motion",
#                 f"{metrics['std_magnitude']:.2f}"
#             )

#         with col4:
#             st.metric(
#                 "Frames",
#                 metrics["total_frames"]
#             )

#         # DOWNLOAD JSON
#         result_json = {
#             "sentiment": result["sentiment"],
#             "confidence": float(result["confidence"]),
#             "probabilities": {
#                 "positive": float(result["probabilities"][0]),
#                 "neutral": float(result["probabilities"][1]),
#                 "negative": float(result["probabilities"][2])
#             },
#             "timestamp": datetime.now().isoformat()
#         }

#         st.download_button(
#             label="📥 Download Result JSON",
#             data=json.dumps(result_json, indent=4),
#             file_name="analysis_result.json",
#             mime="application/json"
#         )

#     except Exception as e:

#         st.error(f"❌ Error: {str(e)}")

#         logger.error(str(e), exc_info=True)

#     finally:

#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)

# # =========================================================
# # SIDEBAR
# # =========================================================

# with st.sidebar:

#     st.header("⚙ Settings")

#     device_option = st.selectbox(
#         "Select Device",
#         [
#             "CUDA (GPU)" if torch.cuda.is_available() else "CPU",
#             "CPU"
#         ]
#     )

#     st.session_state.device = (
#         "cuda"
#         if device_option.startswith("CUDA")
#         else "cpu"
#     )

#     st.subheader("Parameters")

#     max_frames = st.slider(
#         "Max Frames",
#         10,
#         60,
#         30
#     )

#     fps = st.slider(
#         "FPS",
#         5,
#         30,
#         10
#     )

#     st.markdown("---")

#     st.info("""
#     SceneMotion-LLM uses:

#     - ResNet50
#     - Optical Flow
#     - LSTM
#     - Attention
#     - PyTorch
#     """)

# # =========================================================
# # TABS
# # =========================================================

# tab1, tab2, tab3 = st.tabs([
#     "📹 Analyze",
#     "📊 Batch",
#     "ℹ Info"
# ])

# # =========================================================
# # TAB 1
# # =========================================================

# with tab1:

#     st.header("Upload Video")

#     load_model_and_inferencer()

#     uploaded_file = st.file_uploader(
#         "Upload Video",
#         type=[
#             "mp4",
#             "avi",
#             "mov",
#             "mkv"
#         ]
#     )

#     if uploaded_file is not None:

#         st.video(uploaded_file)

#         col1, col2 = st.columns(2)

#         with col1:
#             st.write(
#                 f"**Filename:** {uploaded_file.name}"
#             )

#         with col2:
#             st.write(
#                 f"**Size:** "
#                 f"{uploaded_file.size / 1024 / 1024:.2f} MB"
#             )

#         if st.button(
#             "🚀 Analyze Sentiment",
#             use_container_width=True
#         ):

#             analyze_video_file(uploaded_file)

#     else:

#         st.info("Upload a video to start analysis")

# # =========================================================
# # TAB 2
# # =========================================================

# with tab2:

#     st.header("Batch Processing")

#     st.info(
#         "Batch processing feature coming soon."
#     )

# # =========================================================
# # TAB 3
# # =========================================================

# with tab3:

#     st.header("System Information")

#     st.write(
#         f"**PyTorch Version:** {torch.__version__}"
#     )

#     st.write(
#         f"**CUDA Available:** "
#         f"{torch.cuda.is_available()}"
#     )

#     if torch.cuda.is_available():

#         st.write(
#             f"**GPU:** "
#             f"{torch.cuda.get_device_name(0)}"
#         )

#     st.subheader("Architecture")

#     st.code("""
# SceneMotionLLMModel(
#     spatial_feature_dim=2048,
#     temporal_hidden_dim=256,
#     fusion_dim=512,
#     num_classes=3
# )
# """)

# # =========================================================
# # FOOTER
# # =========================================================

# st.markdown("---")

# st.markdown(
#     "Made with ❤️ using Streamlit + PyTorch"
# )



# """
# Streamlit Web App - SceneMotion-LLM (FIXED VERSION)
# """

# import streamlit as st
# import torch
# import os
# import tempfile
# import json
# import logging
# from datetime import datetime

# from models.sentiment_classifier import SceneMotionLLMModel
# from inference import SceneMotionInferencer

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # =========================
# # CONFIG
# # =========================
# st.set_page_config(
#     page_title="SceneMotion-LLM",
#     page_icon="🎬",
#     layout="wide"
# )

# # =========================
# # SESSION STATE
# # =========================
# if "model" not in st.session_state:
#     st.session_state.model = None

# if "inferencer" not in st.session_state:
#     st.session_state.inferencer = None

# if "device" not in st.session_state:
#     st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"


# # =========================
# # LOAD MODEL
# # =========================
# def load_model():

#     if st.session_state.model is not None:
#         return

#     with st.spinner("Loading model..."):

#         model = SceneMotionLLMModel(
#             spatial_feature_dim=512,
#             motion_feature_dim=2,
#             temporal_hidden_dim=256,
#             attention_dim=128,
#             fusion_dim=512,
#             num_classes=3,
#             max_frames=30,
#             dropout=0.3
#         )

#         checkpoint_path = "./checkpoints/best_model.pt"

#         if os.path.exists(checkpoint_path):
#             checkpoint = torch.load(checkpoint_path, map_location=st.session_state.device)
#             model.load_state_dict(checkpoint["model_state_dict"])
#             st.success("✅ Model loaded from checkpoint")
#         else:
#             st.warning("⚠ No checkpoint found (random weights)")

#         model.to(st.session_state.device)
#         model.eval()

#         st.session_state.model = model

#         st.session_state.inferencer = SceneMotionInferencer(
#             model=model,
#             checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
#             device=st.session_state.device,
#             max_frames=30
#         )


# # =========================
# # DISPLAY RESULT
# # =========================
# def show_result(result):

#     st.subheader("🎯 Prediction")

#     st.markdown(f"### {result['sentiment']}")
#     st.metric("Confidence", f"{result['confidence']:.2%}")

#     st.subheader("📊 Probabilities")

#     labels = ["Positive", "Neutral", "Negative"]

#     for i, label in enumerate(labels):
#         st.write(label)
#         st.progress(float(result["probabilities"][i]))
#         st.write(f"{result['probabilities'][i]:.2%}")

#     st.subheader("🧠 Motion Analysis")

#     st.write(result.get("motion_analysis", "N/A"))

#     st.subheader("📈 Motion Features")

#     mf = result.get("motion_features", {})

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Mean", mf.get("mean_magnitude", 0))
#     col2.metric("Max", mf.get("max_magnitude", 0))
#     col3.metric("Std", mf.get("std_magnitude", 0))

#     st.subheader("📝 AI Report")

#     st.text(result.get("report", "No report generated"))


# # =========================
# # VIDEO ANALYSIS
# # =========================
# def analyze_video(video_file):

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(video_file.read())
#         video_path = tmp.name

#     try:
#         load_model()

#         with st.spinner("Analyzing video..."):

#             result = st.session_state.inferencer.analyze_video(
#                 video_path,
#                 output_dir=tempfile.mkdtemp(),
#                 save_frames=False
#             )

#         st.success("✅ Analysis complete")

#         show_result(result)

#         # DOWNLOAD BUTTON
#         download_data = {
#             "sentiment": result["sentiment"],
#             "confidence": float(result["confidence"]),
#             "probabilities": {
#                 "positive": float(result["probabilities"][0]),
#                 "neutral": float(result["probabilities"][1]),
#                 "negative": float(result["probabilities"][2])
#             },
#             "timestamp": datetime.now().isoformat()
#         }

#         st.download_button(
#             "📥 Download JSON",
#             data=json.dumps(download_data, indent=2),
#             file_name="result.json",
#             mime="application/json"
#         )

#     except Exception as e:
#         st.error(f"Error: {str(e)}")
#         logger.error(str(e), exc_info=True)

#     finally:
#         if os.path.exists(video_path):
#             os.remove(video_path)


# # =========================
# # UI
# # =========================
# st.title("🎬 SceneMotion-LLM")
# st.markdown("Video Sentiment Analysis using Deep Learning")

# uploaded_file = st.file_uploader(
#     "Upload Video",
#     type=["mp4", "avi", "mov", "mkv"]
# )

# if uploaded_file:
#     st.video(uploaded_file)

#     if st.button("🚀 Analyze Video"):
#         analyze_video(uploaded_file)

# else:
#     st.info("Upload a video to start analysis")

# # =========================
# # FOOTER
# # =========================
# st.markdown("---")
# st.markdown("Made with ❤️ using PyTorch + Streamlit")



# """
# SceneMotion-LLM - Modern Streamlit UI (Human-Like Design)
# """

# import streamlit as st
# import torch
# import os
# import tempfile
# import json
# from datetime import datetime

# from models.sentiment_classifier import SceneMotionLLMModel
# from inference import SceneMotionInferencer


# # =========================
# # PAGE CONFIG
# # =========================
# st.set_page_config(
#     page_title="SceneMotion AI",
#     page_icon="🎬",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # =========================
# # MODERN UI CSS
# # =========================
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

# html, body, [class*="css"] {
#     font-family: 'Inter', sans-serif;
#     background-color: #0b1220;
#     color: #ffffff;
# }

# /* Title */
# h1 {
#     font-size: 42px !important;
#     text-align: center;
#     font-weight: 700;
#     background: linear-gradient(90deg, #00c6ff, #0072ff);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     margin-bottom: 0px;
# }

# /* Subtitle */
# .subtitle {
#     text-align: center;
#     color: #94a3b8;
#     font-size: 18px;
#     margin-bottom: 30px;
# }

# /* Cards */
# .card {
#     background-color: #111827;
#     padding: 20px;
#     border-radius: 16px;
#     border: 1px solid #1f2937;
#     margin-bottom: 15px;
# }

# /* Buttons */
# div.stButton > button {
#     background: linear-gradient(90deg, #0072ff, #00c6ff);
#     color: white;
#     border-radius: 12px;
#     padding: 0.7rem 1.2rem;
#     border: none;
#     font-size: 16px;
#     font-weight: 600;
# }

# div.stButton > button:hover {
#     transform: scale(1.02);
#     box-shadow: 0 0 15px rgba(0, 114, 255, 0.5);
# }

# /* File uploader */
# [data-testid="stFileUploader"] {
#     border: 2px dashed #334155;
#     background-color: #0f172a;
#     padding: 18px;
#     border-radius: 12px;
# }

# /* Metrics */
# [data-testid="stMetric"] {
#     background-color: #111827;
#     border-radius: 12px;
#     padding: 10px;
#     border: 1px solid #1f2937;
# }

# /* Progress */
# .stProgress > div > div > div {
#     background: linear-gradient(90deg, #00c6ff, #0072ff);
# }
# </style>
# """, unsafe_allow_html=True)


# # =========================
# # TITLE
# # =========================
# st.markdown("# 🎬 SceneMotion AI")
# st.markdown('<div class="subtitle">Video Emotion • Motion Understanding • AI Analysis Engine</div>', unsafe_allow_html=True)


# # =========================
# # SESSION STATE
# # =========================
# if "model" not in st.session_state:
#     st.session_state.model = None

# if "inferencer" not in st.session_state:
#     st.session_state.inferencer = None

# if "device" not in st.session_state:
#     st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"


# # =========================
# # LOAD MODEL
# # =========================
# def load_model():

#     if st.session_state.model is not None:
#         return

#     with st.spinner("Loading AI Model..."):

#         model = SceneMotionLLMModel(
#             spatial_feature_dim=512,
#             motion_feature_dim=2,
#             temporal_hidden_dim=256,
#             attention_dim=128,
#             fusion_dim=512,
#             num_classes=3,
#             max_frames=30,
#             dropout=0.3
#         )

#         checkpoint_path = "./checkpoints/best_model.pt"

#         if os.path.exists(checkpoint_path):
#             checkpoint = torch.load(checkpoint_path, map_location=st.session_state.device)
#             model.load_state_dict(checkpoint["model_state_dict"])
#             st.success("✅ Pretrained Model Loaded")
#         else:
#             st.warning("⚠ Running with random weights")

#         model.to(st.session_state.device)
#         model.eval()

#         st.session_state.model = model

#         st.session_state.inferencer = SceneMotionInferencer(
#             model=model,
#             checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
#             device=st.session_state.device,
#             max_frames=30
#         )


# # =========================
# # RESULT UI
# # =========================
# def show_result(result):

#     st.markdown("## 🧠 AI Analysis Result")

#     st.markdown(f"""
#     <div class="card">
#         <h3>🎯 Sentiment: {result['sentiment']}</h3>
#         <p><b>Confidence:</b> {result['confidence']:.2%}</p>
#     </div>
#     """, unsafe_allow_html=True)

#     st.markdown("### 📊 Probability Distribution")

#     labels = ["Positive", "Neutral", "Negative"]

#     for i, label in enumerate(labels):
#         st.write(label)
#         st.progress(float(result["probabilities"][i]))

#     col1, col2, col3 = st.columns(3)

#     mf = result.get("motion_features", {})

#     col1.metric("Mean Motion", f"{mf.get('mean_magnitude', 0):.2f}")
#     col2.metric("Max Motion", f"{mf.get('max_magnitude', 0):.2f}")
#     col3.metric("Std Motion", f"{mf.get('std_magnitude', 0):.2f}")

#     st.markdown("### 🧾 AI Generated Report")

#     st.markdown(f"""
#     <div class="card">
#         {result.get('report', 'No report available')}
#     </div>
#     """, unsafe_allow_html=True)


# # =========================
# # ANALYSIS FUNCTION
# # =========================
# def analyze_video(video_file):

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#         tmp.write(video_file.read())
#         path = tmp.name

#     try:
#         load_model()

#         with st.spinner("AI is analyzing your video..."):

#             result = st.session_state.inferencer.analyze_video(
#                 path,
#                 output_dir=tempfile.mkdtemp(),
#                 save_frames=False
#             )

#         st.success("Analysis Complete 🎉")

#         show_result(result)

#         # DOWNLOAD
#         st.download_button(
#             "📥 Download JSON Report",
#             json.dumps({
#                 "sentiment": result["sentiment"],
#                 "confidence": float(result["confidence"]),
#                 "probabilities": {
#                     "positive": float(result["probabilities"][0]),
#                     "neutral": float(result["probabilities"][1]),
#                     "negative": float(result["probabilities"][2])
#                 },
#                 "timestamp": datetime.now().isoformat()
#             }, indent=2),
#             file_name="scene_analysis.json",
#             mime="application/json"
#         )

#     finally:
#         if os.path.exists(path):
#             os.remove(path)


# # =========================
# # UPLOAD SECTION
# # =========================
# st.markdown("## 📤 Upload Video")

# uploaded_file = st.file_uploader(
#     "Drag & drop or browse video file",
#     type=["mp4", "avi", "mov", "mkv"]
# )

# if uploaded_file:
#     st.video(uploaded_file)

#     st.markdown("### Ready for AI Analysis 🚀")

#     if st.button("Analyze Video"):
#         analyze_video(uploaded_file)

# else:
#     st.info("Upload a video to start AI analysis")


# # =========================
# # FOOTER
# # =========================
# st.markdown("---")
# st.markdown("Made with ❤️ using PyTorch + Streamlit + AI")





import streamlit as st
import torch
import tempfile
import os
import json
from datetime import datetime

from models.sentiment_classifier import SceneMotionLLMModel
from inference import SceneMotionInferencer

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SceneMotion AI",
    page_icon="🎬",
    layout="wide"
)

# =========================================================
# MODERN GLASS UI
# =========================================================
st.markdown("""
<style>

html, body {
    font-family: 'Segoe UI', sans-serif;
    background: #0b1220;
    color: #ffffff;
}

.main {
    background: transparent;
}

/* Glass card */
.glass {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Title */
.title {
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    color: #ffffff;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #a5b4fc;
    margin-bottom: 20px;
}

/* Sentiment styles */
.pos { color: #22c55e; font-size: 26px; font-weight: bold; }
.neu { color: #facc15; font-size: 26px; font-weight: bold; }
.neg { color: #ef4444; font-size: 26px; font-weight: bold; }

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">🎬 SceneMotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Video Sentiment Analysis (Deep Learning)</div>', unsafe_allow_html=True)

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

            checkpoint_path = "./checkpoints/best_model.pt"

            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=st.session_state.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                st.success("✅ Model Loaded")
            else:
                st.warning("⚠ No checkpoint found (using random weights)")

            model.to(st.session_state.device)
            model.eval()

            st.session_state.model = model

            st.session_state.inferencer = SceneMotionInferencer(
                model=model,
                checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
                device=st.session_state.device,
                max_frames=30
            )

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

    st.subheader("📊 Probability")
    labels = ["Positive", "Neutral", "Negative"]

    for i in range(3):
        st.write(f"{labels[i]}")
        st.progress(float(probs[i]))
        st.write(f"{probs[i]:.2%}")

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

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🧠 AI Summary")

        # NO CRASH SAFE OUTPUT
        if "motion_analysis" in result:
            st.write(result["motion_analysis"])
        else:
            st.info("Motion analysis not available in current model output.")

        if "report" in result:
            st.text(result["report"])
        else:
            st.info("Detailed report not generated.")

        st.markdown("</div>", unsafe_allow_html=True)

        # DOWNLOAD
        download_data = {
            "sentiment": result.get("sentiment"),
            "confidence": float(result.get("confidence", 0)),
            "probabilities": result.get("probabilities", []).tolist() if hasattr(result.get("probabilities"), "tolist") else result.get("probabilities"),
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
            os.remove(path)

# =========================================================
# UI SECTION
# =========================================================
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📤 Upload Video")

    file = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv"])

    if file:
        st.video(file)

        if st.button("🚀 Analyze Video"):
            analyze_video(file)

    else:
        st.info("Upload a video to start analysis")

    st.markdown("</div>", unsafe_allow_html=True)