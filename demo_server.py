"""
Live inference server for SceneMotion AI.
Accepts a video upload, extracts CLIP + audio + motion features, returns sentiment.

Usage:
  pip install flask
  python demo_server.py
Then open http://localhost:5050 in your browser.
"""
import sys, json, tempfile, os
from pathlib import Path

import numpy as np
import torch
import joblib
import cv2
import open_clip
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(r".\scene_motion_llm\checkpoints\clip_fusion\clip_mlp_model.joblib")
PILLAR_CACHE = Path(r".\scene_motion_llm\cache\semantic_five_pillar_v2")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLED_FRAMES = 16
MAX_DURATION   = 15.0
CLASS_NAMES    = ["Negative", "Neutral", "Positive"]

# ── Load CLIP once at startup ────────────────────────────────────────────────
print(f"Loading CLIP ViT-B/32 on {DEVICE}...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(DEVICE).eval()
print("CLIP loaded.")

# ── Load MLP model ───────────────────────────────────────────────────────────
print(f"Loading MLP from {MODEL_PATH}...")
mlp = joblib.load(MODEL_PATH)
print("MLP loaded.")

app = Flask(__name__, static_folder=".")

# ── Feature extraction helpers ───────────────────────────────────────────────
def extract_clip_features(video_path: str) -> np.ndarray:
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
            frames.append(clip_preprocess(Image.fromarray(rgb)))
    cap.release()
    if not frames:
        return np.zeros(512, dtype=np.float32)
    with torch.no_grad():
        tensors = torch.stack(frames).to(DEVICE)
        feats = clip_model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu().float().numpy()
    return feats.mean(axis=0)  # (512,)


def extract_audio_features(video_path: str) -> np.ndarray:
    """Extract basic MFCC-like features using librosa if available, else zeros."""
    try:
        import librosa
        y, sr = librosa.load(video_path, sr=22050, duration=MAX_DURATION, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        return mfcc.mean(axis=1).astype(np.float32)
    except Exception:
        return np.zeros(20, dtype=np.float32)


def extract_motion_features(video_path: str) -> np.ndarray:
    """Extract optical flow magnitude stats (14-dim)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, int(fps * MAX_DURATION))
    indices = np.linspace(0, max_frame, SAMPLED_FRAMES + 1, dtype=int)
    grays = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(grays) < 2:
        return np.zeros(14, dtype=np.float32)
    mags = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(mag.flatten())
    mags = np.array(mags)  # (N, H*W)
    feat = np.array([
        mags.mean(), mags.std(), mags.max(),
        np.percentile(mags, 25), np.percentile(mags, 75),
        mags.mean(axis=1).max(), mags.mean(axis=1).min(),
        mags.std(axis=1).mean(), mags.std(axis=1).std(),
        (mags > mags.mean()).mean(),
        mags.mean(axis=0).max(), mags.mean(axis=0).std(),
        mags.max(axis=1).mean(), mags.max(axis=1).std(),
    ], dtype=np.float32)
    return feat


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "demo.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    suffix = Path(video_file.filename).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        clip_feat  = extract_clip_features(tmp_path)
        audio_feat = extract_audio_features(tmp_path)
        motion_feat = extract_motion_features(tmp_path)

        X = np.concatenate([clip_feat, audio_feat, motion_feat]).reshape(1, -1)
        probs  = mlp.predict_proba(X)[0]
        pred   = int(probs.argmax())
        label  = CLASS_NAMES[pred]

        return jsonify({
            "label": label,
            "confidence": float(probs[pred]),
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("\n🎬 SceneMotion AI demo running at http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
