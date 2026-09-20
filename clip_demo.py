"""
Streamlit demo — uses the CLIP MLP model for live video sentiment.
Fits the MLP from cached features on first run (~30s). No pre-saved model needed.

Run:
  .\.venv\Scripts\python.exe -m streamlit run clip_demo.py
"""
import os, tempfile
import numpy as np
import torch
import cv2
import open_clip
from pathlib import Path
from PIL import Image
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# ── Constants ─────────────────────────────────────────────────────────────────
CLIP_CACHE   = Path(r".\scene_motion_llm\cache\clip_vitb32_v1")
PILLAR_CACHE = Path(r".\scene_motion_llm\cache\semantic_five_pillar_v2")
RANKING_PATH = Path(r".\scene_motion_llm\dataset\annotations\ACCEDEranking.txt")
SETS_PATH    = Path(r"C:\Users\student\Downloads\LIRIS-ACCEDE-annotations\LIRIS-ACCEDE-annotations\annotations\ACCEDEsets.txt")
CLASS_NAMES  = ["Negative", "Neutral", "Positive"]
EMOJIS       = {"Positive": "😊", "Neutral": "😐", "Negative": "😟"}
COLORS       = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLED_FRAMES = 16
MAX_DURATION   = 15.0

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SceneMotion AI", page_icon="🎬", layout="centered")
st.markdown("""
<style>
.stApp { background: #0f172a; color: #e2e8f0; }
.block-container { max-width: 680px; padding-top: 2rem; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, label, .stCaption { color: #94a3b8 !important; }
</style>""", unsafe_allow_html=True)

st.title("🎬 SceneMotion AI")
st.caption("CLIP ViT-B/32 + MLP — 43.95% macro-F1 on LIRIS-ACCEDE")

# ── Load labels from cache ────────────────────────────────────────────────────
@st.cache_data
def load_labels():
    rankings = {}
    with open(RANKING_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                vid = parts[1].replace(".mp4", "")
                rankings[vid] = int(parts[2])

    total = 9800
    split_map = {}
    with open(SETS_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                vid = parts[1].replace(".mp4", "")
                code = int(parts[2])
                split = {1: "train", 2: "validation", 0: "test"}.get(code)
                if split:
                    split_map[vid] = split

    # Assign labels by rank thirds
    labels = {s: {} for s in ["train", "validation", "test"]}
    for vid, split in split_map.items():
        rank = rankings.get(vid)
        if rank is not None:
            if rank <= 3265:
                lbl = 0   # Negative
            elif rank <= 6532:
                lbl = 1   # Neutral
            else:
                lbl = 2   # Positive
            labels[split][vid] = lbl
    return labels

# ── Build feature matrix from cache ──────────────────────────────────────────
@st.cache_data
def build_features_for_split(split):
    labels = load_labels()
    X, y = [], []
    for vid, lbl in labels[split].items():
        clip_npz = CLIP_CACHE / split / f"{vid}.npz"
        if not clip_npz.exists():
            continue
        clip_feat = np.load(clip_npz)["embeddings"].mean(axis=0)

        audio_npz = PILLAR_CACHE / split / f"{vid}.npz"
        if audio_npz.exists():
            d = np.load(audio_npz)
            audio = d["feature_audio"].mean(axis=0) if "feature_audio" in d else np.zeros(20)
            motion = d["feature_motion"].mean(axis=0) if "feature_motion" in d else np.zeros(14)
        else:
            audio = np.zeros(20)
            motion = np.zeros(14)

        X.append(np.concatenate([clip_feat, audio, motion]).astype(np.float32))
        y.append(lbl)
    return np.array(X), np.array(y)

def build_training_features():
    return build_features_for_split("train")

# ── Fit MLP (cached so it only trains once per session) ──────────────────────
@st.cache_resource
def get_mlp():
    X_train, y_train = build_training_features()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(512,), alpha=0.001,
                              max_iter=500, early_stopping=True, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    return pipe

# ── Load CLIP ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_clip():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return model.to(DEVICE).eval(), preprocess

# ── Feature extraction from uploaded video ────────────────────────────────────
def extract_features(video_path):
    model, preprocess = get_clip()

    # CLIP frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = min(total - 1, int(fps * MAX_DURATION))
    indices = np.linspace(0, max_frame, SAMPLED_FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(preprocess(Image.fromarray(rgb)))
    cap.release()

    if not frames:
        clip_feat = np.zeros(512, dtype=np.float32)
    else:
        with torch.no_grad():
            t = torch.stack(frames).to(DEVICE)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        clip_feat = f.cpu().float().numpy().mean(axis=0)

    # Basic audio (zeros if librosa not available)
    try:
        import librosa
        y, sr = librosa.load(video_path, sr=22050, duration=MAX_DURATION, mono=True)
        audio = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1).astype(np.float32)
    except Exception:
        audio = np.zeros(20, dtype=np.float32)

    # Basic motion
    cap2 = cv2.VideoCapture(video_path)
    max_f2 = min(int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, int((cap2.get(cv2.CAP_PROP_FPS) or 25) * MAX_DURATION))
    idxs2 = np.linspace(0, max_f2, SAMPLED_FRAMES + 1, dtype=int)
    grays = []
    for idx in idxs2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap2.read()
        if ok:
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap2.release()

    if len(grays) >= 2:
        mags = []
        for i in range(len(grays) - 1):
            flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append(mag.flatten())
        mags = np.array(mags)
        motion = np.array([mags.mean(), mags.std(), mags.max(),
                           np.percentile(mags,25), np.percentile(mags,75),
                           mags.mean(1).max(), mags.mean(1).min(),
                           mags.std(1).mean(), mags.std(1).std(),
                           (mags>mags.mean()).mean(),
                           mags.mean(0).max(), mags.mean(0).std(),
                           mags.max(1).mean(), mags.max(1).std()], dtype=np.float32)
    else:
        motion = np.zeros(14, dtype=np.float32)

    return np.concatenate([clip_feat, audio, motion]).reshape(1, -1)

# ── UI ────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Drop a video clip", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    st.video(uploaded)
    if st.button("✨ Analyze Sentiment", type="primary", use_container_width=True):
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        try:
            with st.spinner("Loading CLIP + fitting model from cache (~30s first time)..."):
                mlp = get_mlp()

            with st.spinner("Extracting features from your video..."):
                X = extract_features(tmp_path)

            raw_probs = mlp.predict_proba(X)[0]
            # Temperature scaling — softens overconfident MLP without collapsing to prior
            T = 3.0
            logits = np.log(raw_probs + 1e-10)
            logits_t = logits / T
            exp_l = np.exp(logits_t - logits_t.max())
            probs = exp_l / exp_l.sum()
            pred  = int(probs.argmax())
            label = CLASS_NAMES[pred]
            conf  = float(probs[pred]) * 100

            st.markdown("---")
            color = COLORS[label]
            emoji = EMOJIS[label]
            st.markdown(
                f"<h2 style='text-align:center;color:{color};'>{emoji} {label}</h2>"
                f"<p style='text-align:center;color:#64748b;'>Confidence: {conf:.1f}%</p>",
                unsafe_allow_html=True)

            st.markdown("**Scores:**")
            for cls in CLASS_NAMES:
                i = CLASS_NAMES.index(cls)
                st.progress(float(probs[i]), text=f"{EMOJIS[cls]} {cls}: {probs[i]*100:.1f}%")

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.unlink(tmp_path)
