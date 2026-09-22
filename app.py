r"""
SceneMotion AI — Six-Signal Heuristic Video Analyzer

A local multimodal video-analysis demonstrator. It combines pretrained CLIP,
Whisper, and DistilBERT models with deterministic audiovisual cues to produce a
heuristic Negative/Neutral/Positive score; it is not a clinically validated
psychological assessment or a trained end-to-end sentiment predictor.

  1. CLIP Zero-Shot       (0.40) — Spatial prompt-similarity signal
  2. Speech (Whisper+DistilBERT) (0.10) — Speech transcription and text-emotion signal
  3. Acoustic (Librosa)   (0.25) — Hand-crafted prosody signal
  4. Color (HSV)          (0.05) — Brightness and hue heuristic
  5. Motion (Optical Flow)(0.15) — Explicitly abstains from valence fusion
  6. Temporal (Shot Cuts) (0.05) — Explicitly abstains from valence fusion

Run:
  .\.codex-venv\Scripts\python.exe -m streamlit run app.py
"""
import os
import tempfile
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import open_clip
import librosa
import streamlit as st
from PIL import Image

try:
    # MoviePy 2 exposes VideoFileClip at the package root.
    from moviepy import VideoFileClip
except ImportError:
    # The project currently pins MoviePy 1.x, which exposes it here.
    from moviepy.editor import VideoFileClip

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Negative", "Neutral", "Positive"]
EMOJIS       = {"Positive": "😊", "Neutral": "😐", "Negative": "😟"}
COLORS       = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DURATION = 15.0
SAMPLED_FRAMES = 16

# Heuristic base weights.  A usable pillar keeps its configured contribution;
# model confidence does not increase that contribution.
# They are product choices, not a validated translation of Mehrabian's 7-38-55
# study into video-sentiment model weights.
CLIP_W    = 0.40   # Visual prompt-similarity signal
SPEECH_W  = 0.10   # Transcribed speech / text-emotion signal
ACOUS_W   = 0.25   # Hand-crafted prosody signal
COLOR_W   = 0.05   # Brightness / hue heuristic
MOTION_W  = 0.15   # Optical-flow heuristic
TEMP_W    = 0.05   # Editing-style heuristic

# These scores are relative fusion scores, not probabilities.  Only a near tie
# should be presented as mixed; three-way scores do not need to exceed 50% to
# have a meaningful leading class.
MIXED_MARGIN = 0.05
MIXED_PAIR_TIE_LABELS = {
    frozenset(("Negative", "Neutral")): "Mixed: Negative / Neutral",
    frozenset(("Negative", "Positive")): "Mixed: Negative / Positive",
    frozenset(("Neutral", "Positive")): "Mixed: Neutral / Positive",
}
MIXED_ALL_LABEL = "Mixed: Negative / Neutral / Positive"

# ── CLIP prompt groups ───────────────────────────────────────────────────────
# These terms draw on common emotion taxonomies, but are only zero-shot text
# prompts—not evidence that the system recognizes or diagnoses those emotions.
# The DistilBERT speech model emits sadness, joy, love, anger, fear, and
# surprise; the visual prompt groups intentionally also include calm, disgust,
# and everyday scenes.
TEXT_PROMPTS = {
    "Negative": [
        "a video scene conveying sadness",
        "a video scene conveying fear",
        "a video scene conveying anger",
        "a video scene conveying disgust"  # Ekman's 6th emotion
    ],
    "Neutral": [
        "a video scene conveying calm",
        "a video scene conveying surprise",
        "a neutral everyday scene"
    ],
    "Positive": [
        "a video scene conveying joy",
        "a video scene conveying love",
        "a video scene conveying happiness"
    ],
}

SPEECH_STATUS_MARKERS = (
    "[No audio",
    "[Audio too short]",
    "[No speech detected]",
    "[Gibberish Detected",
    "[Low Confidence: Muted]",
    "[Error:",
)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SceneMotion AI", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.stApp { background: #0f172a; color: #e2e8f0; }
.block-container { max-width: 1000px; padding-top: 2rem; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, label, .stCaption { color: #94a3b8 !important; }
</style>""", unsafe_allow_html=True)

st.title("🎬 SceneMotion AI")
st.caption("Six-signal heuristic fusion · Spatial (CLIP) · Speech (DistilBERT) · Acoustic (Librosa) · Color (HSV) · Motion (Optical Flow) · Temporal (Shot Cuts) · Scores are not calibrated confidence estimates")

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_clip():
    # Pretrained weights may be downloaded on first use when absent from cache.
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model.to(DEVICE).eval(), preprocess, tokenizer

@st.cache_resource
def load_stt_model():
    import whisper
    # Whisper's pretrained weights may be downloaded on first use.
    return whisper.load_model("base", device=DEVICE)

@st.cache_resource
def load_emotion_model():
    from transformers import pipeline
    # Fine-tuned Twitter sentiment classifier; model files may download on first
    # use and are cached locally afterwards. This is highly robust to casual text.
    return pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None)


# ── Extraction using MoviePy ─────────────────────────────────────────────────
def extract_media(video_path: str):
    """
    Extract sampled frames and, when present, a matching temporary WAV segment.

    MoviePy delegates decoding to its ffmpeg integration. It avoids relying on
    OpenCV for audio extraction, but still requires a usable MoviePy/ffmpeg
    installation. Both signals are limited to the same initial time window.
    """
    clip = VideoFileClip(video_path)
    wav_path = None
    try:
        duration = float(clip.duration or 0.0)
        dur = min(duration, MAX_DURATION)
        fps = clip.fps if clip.fps and clip.fps > 0 else 25.0

        # Extract frames.
        total_frames = int(fps * dur)
        if total_frames <= 0:
            frames = []
        else:
            indices = np.linspace(0, total_frames - 1, SAMPLED_FRAMES, dtype=int)
            frames = [clip.get_frame(idx / fps) for idx in indices]

        # The source upload path is a unique NamedTemporaryFile path, so its
        # companion WAV cannot collide with another analysis invocation. Trim
        # the soundtrack to the frame-analysis window to keep both signals
        # aligned and bound transcription work on long uploads.
        audio = clip.audio
        if audio is not None and dur > 0:
            wav_path = video_path + "_audio.wav"
            if hasattr(audio, "subclipped"):
                audio_segment = audio.subclipped(0, dur)  # MoviePy 2
            else:
                audio_segment = audio.subclip(0, dur)  # MoviePy 1
            try:
                audio_segment.write_audiofile(wav_path, fps=22050, logger=None)
            finally:
                if audio_segment is not audio:
                    audio_segment.close()

        return frames, wav_path
    except Exception:
        # A failed extraction can leave a partial WAV behind; remove it before
        # propagating the original error to the UI.
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        raise
    finally:
        clip.close()


def speech_has_status(transcript: str) -> bool:
    """Whether text contains an availability or abstention status, not speech."""
    return any(marker in transcript for marker in SPEECH_STATUS_MARKERS)


# ── Signal 1: CLIP zero-shot ─────────────────────────────────────────────────
def run_clip_zeroshot(frames) -> np.ndarray:
    if not frames:
        print("CLIP Warning: No frames provided")
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    clip_model, preprocess, tokenizer = load_clip()

    processed = []
    for f in frames:
        processed.append(preprocess(Image.fromarray(f)))

    with torch.no_grad():
        img = clip_model.encode_image(torch.stack(processed).to(DEVICE))
        img = img / img.norm(dim=-1, keepdim=True) # [frames, dim]

        frame_scores_per_class = []
        for cls in CLASS_NAMES:
            tokens   = tokenizer(TEXT_PROMPTS[cls]).to(DEVICE)
            txt      = clip_model.encode_text(tokens)
            txt      = txt / txt.norm(dim=-1, keepdim=True) # [prompts, dim]

            # Score each frame against all prompts for this class
            # img: [F, D], txt: [P, D] -> [F, P]
            sim = img @ txt.T
            # For each frame, get the best matching prompt in this class
            best_prompt_per_frame = sim.max(dim=1).values # [F]
            # Average the best scores across all frames
            class_score = best_prompt_per_frame.mean().item()
            frame_scores_per_class.append(class_score)

    s = np.array(frame_scores_per_class) - max(frame_scores_per_class)
    return (np.exp(s * 10.0) / np.exp(s * 10.0).sum()).astype(np.float32) # Temperature scaling to boost confidence


# ── Signal 2: Speech transcription + emotion classifier ──────────────────────
def run_speech_sentiment(wav_path: str):
    if not wav_path or not os.path.exists(wav_path):
        return np.array([1/3, 1/3, 1/3], dtype=np.float32), "[No audio track found]"

    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        if len(y) < sr * 0.5:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32), "[Audio too short]"

        model = load_stt_model()
        result = model.transcribe(wav_path, fp16=False)
        transcription = result.get("text", "").lower().strip()

        if not transcription.strip():
            return np.array([1/3, 1/3, 1/3], dtype=np.float32), "[No speech detected]"

        # Lexical Validity Filter (Gibberish Check)
        from spellchecker import SpellChecker
        import re
        spell = SpellChecker()
        # Include apostrophes to prevent contractions (like "wasn't") from being split into "wasn" and "t"
        words = re.findall(r"[a-z']+", transcription)
        # Strip trailing apostrophes (e.g. from quotes) but keep internal ones
        words = [w.strip("'") for w in words if w.strip("'")]
        if len(words) > 0:
            known_words = spell.known(words)
            validity_score = len(known_words) / len(words)
            if validity_score < 0.60:
                return np.array([1/3, 1/3, 1/3], dtype=np.float32), transcription + f" [Gibberish Detected — Muting ({validity_score*100:.0f}% lexical validity)]"

        emotion_classifier = load_emotion_model()
        # Returns a list of dicts like: [{'label': 'joy', 'score': 0.9}, ...]
        scores = emotion_classifier(transcription)[0]

        # Initialize our 3 sentiment buckets
        neg_score = 0.0
        pos_score = 0.0
        neu_score = 0.0

        # Map RoBERTa's native labels to our 3 Sentiment classes
        for s in scores:
            label = s['label']
            val = s['score']
            if label == 'negative':
                neg_score += val
            elif label == 'positive':
                pos_score += val
            elif label == 'neutral':
                neu_score += val

        # Abstain when the classifier has no strong bucket preference. This is a
        # heuristic confidence gate, not a reliable detector of gibberish.
        # Include Neutral here: surprise is the model's neutral-mapped label.
        if max(neg_score, neu_score, pos_score) < 0.50:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32), transcription + " [Low Confidence: Muted]"

        raw = np.array([neg_score, neu_score, pos_score], dtype=np.float32)
        raw = np.clip(raw, 1e-3, None)
        return raw / raw.sum(), transcription

    except Exception as e:
        print(f"Error in STT: {e}")
        return np.array([1/3, 1/3, 1/3], dtype=np.float32), f"[Error: {e}]"


# ── Signal 3: Librosa acoustic emotion ───────────────────────────────────────
def run_acoustic_emotion(wav_path: str) -> np.ndarray:
    if not wav_path or not os.path.exists(wav_path):
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        if len(y) < sr * 0.5:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_mean, rms_std = float(np.mean(rms)), float(np.std(rms))
        
        # Abstain if audio is essentially complete silence
        if rms_mean < 1e-3:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)
            
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
        
        # Wind and unvoiced white noise have exceptionally high zero-crossing rates.
        # If it's noisy and not loud, it's just background room tone or wind.
        if np.mean(zcr) > 0.15 and rms_mean < 0.05:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)
            
        rms_cv = rms_std / (rms_mean + 1e-6)

        try:
            f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
            f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
        except Exception:
            f0_voiced = np.array([])

        if len(f0_voiced) > 5:
            pitch_std, pitch_mean = float(np.std(f0_voiced)), float(np.mean(f0_voiced))
            pitch_cv = pitch_std / (pitch_mean + 1e-6)
            pitch_high = float(np.mean(f0_voiced > 300))
        else:
            pitch_cv, pitch_high = 0.0, 0.0

        spec = np.abs(librosa.stft(y, hop_length=512))
        flux = float(np.clip(np.log1p(np.mean(np.diff(spec, axis=1) ** 2)) / 10.0, 0, 1))

        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rhythmic = float(np.clip(float(tempo) / 180.0, 0, 1)) if not np.isnan(float(tempo)) else 0.3
        except Exception:
            rhythmic = 0.3

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bright = float(np.clip(np.mean(centroid) / 4000.0, 0, 1))

        # Harmonic/percussive ratio is a simple suppression heuristic, not a
        # reliable music or noise classifier.
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic**2) + 1e-6
        percussive_energy = np.sum(y_percussive**2) + 1e-6
        is_musical = harmonic_energy > (percussive_energy * 1.5)

        distress = np.clip(0.35 * pitch_cv * 2.0 + 0.25 * rms_cv + 0.20 * flux + 0.20 * pitch_high, 0, 1)
        if is_musical:
            distress = distress * 0.3

        energy_pos = np.clip(0.40 * rhythmic + 0.30 * bright + 0.30 * rms_mean * 10.0, 0, 1)
        neutral = float(np.clip(1.0 - distress - energy_pos * 0.5, 0, 1))

        raw = np.array([distress, neutral, energy_pos], dtype=np.float32)
        raw = np.clip(raw, 1e-3, None)
        return raw / raw.sum()

    except Exception as e:
        print(f"Error in Acoustic: {e}")
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)


# ── Signal 4: Color Pillar ────────────────────────────────────────────────────
def run_color(frames) -> np.ndarray:
    if not frames:
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    darkness_scores = []
    redness_scores = []
    brightness_scores = []
    greenness_scores = []
    blueness_scores = []
    yellowness_scores = []

    saturations = []
    valid_hues = []

    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        saturations.append(np.mean(s))
        valid_pixels = (s > 20) & (v > 20)
        if np.any(valid_pixels):
            valid_hues.extend(h[valid_pixels].flatten())

        # Darkness = inverse of Value
        dark = 1.0 - (np.mean(v) / 255.0)
        darkness_scores.append(dark)

        # Brightness = Value
        bright = np.mean(v) / 255.0
        brightness_scores.append(bright)

        # Redness (Hue around 0 or 170-180 in OpenCV)
        red_mask = ((h < 10) | (h > 170)) & (s > 100) & (v > 50)
        red_ratio = np.sum(red_mask) / (h.size + 1e-6)
        redness_scores.append(red_ratio)

        # Greenness (Hue around 35-85 in OpenCV)
        green_mask = (h > 35) & (h < 85) & (s > 50) & (v > 50)
        green_ratio = np.sum(green_mask) / (h.size + 1e-6)
        greenness_scores.append(green_ratio)

        # Blueness (Hue around 100-140 in OpenCV)
        blue_mask = (h > 100) & (h < 140) & (s > 50) & (v > 50)
        blue_ratio = np.sum(blue_mask) / (h.size + 1e-6)
        blueness_scores.append(blue_ratio)

        # Yellowness (Hue around 20-35 in OpenCV)
        yellow_mask = (h >= 20) & (h <= 35) & (s > 50) & (v > 50)
        yellow_ratio = np.sum(yellow_mask) / (h.size + 1e-6)
        yellowness_scores.append(yellow_ratio)

    # --- Aesthetic-filter guardrail ---
    avg_sat = np.mean(saturations)
    is_bw = avg_sat < 15
    hue_std = np.std(valid_hues) if len(valid_hues) > 100 else 0
    is_filtered = hue_std < 25 and not is_bw

    if is_bw or is_filtered:
        # A monochrome or strongly filtered clip has limited color variation, so
        # this heuristic abstains instead of treating the palette as valence.
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    avg_dark = float(np.mean(darkness_scores))
    avg_bright = float(np.mean(brightness_scores))
    avg_red = float(np.mean(redness_scores)) * 10.0 # Boost red visibility
    avg_green = float(np.mean(greenness_scores)) * 10.0
    avg_blue = float(np.mean(blueness_scores)) * 10.0
    avg_yellow = float(np.mean(yellowness_scores)) * 10.0

    # Heuristic color vote from brightness and darkness alone.
    base_neg = np.clip(avg_dark * 0.7, 0, 1)
    base_pos = np.clip(avg_bright, 0, 1)
    base_neu = float(np.clip(1.0 - base_neg - base_pos, 0, 1))

    # Do not condition a color vote on CLIP.  That would count the spatial
    # signal twice, rather than treating color as a separate weak cue.
    if avg_red > 0.1:
        base_neu = np.clip(base_neu + avg_red * 0.25, 0, 1)

    # Green is a project-specific negative adjustment.
    if avg_green > 0.1:
        base_neg = np.clip(base_neg + avg_green, 0, 1)

    # Yellow is a project-specific positive adjustment.
    if avg_yellow > 0.1:
        base_pos = np.clip(base_pos + avg_yellow, 0, 1)

    # Blue is a project-specific neutralizing adjustment.
    if avg_blue > 0.2:
        suppression = np.clip(avg_blue, 0, 1)
        base_neg = base_neg * (1.0 - suppression)
        base_pos = base_pos * (1.0 - suppression)
        base_neu = base_neu + suppression

    raw = np.array([base_neg, base_neu, base_pos], dtype=np.float32)
    raw = np.clip(raw, 1e-3, None)
    return raw / raw.sum()

# ── Signal 5: Motion cue (optical-flow heuristic) ──────────────────────────
def run_motion(frames) -> np.ndarray:
    """Abstain from valence classification based on optical flow alone.

    Camera motion, cuts, and a person's movement are not dependable evidence
    for positive or negative sentiment.  Earlier versions converted those
    measurements into a negative vote and blended in CLIP's valence, which both
    created false negatives and counted the spatial signal twice.  Motion may
    remain useful as a future, separately trained activity feature, but it does
    not vote on sentiment in this rule-based demo.
    """
    return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

# ── Signal 6: Temporal cue (editing-style heuristic) ─────────────────────────
def run_temporal(frames) -> np.ndarray:
    """Abstain from valence classification based on editing style alone.

    Cuts, fades, and tilted geometry are filmmaking choices, not reliable
    sentiment labels.  Keeping this explicit abstention is safer than treating
    a fast-cut or low-light clip as negative.
    """
    return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)


def fuse_pillars(pillars: List[Tuple[np.ndarray, float, str]]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fuse signal distributions and return scores plus normalized weights.

    A concentrated model output is not evidence that the model is correct, so
    confidence never increases a pillar's configured contribution.  A uniform
    distribution is an explicit abstention and is removed; remaining base
    weights are then normalized to preserve a valid weighted average.
    """
    effective_weights = []

    uniform = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    for probs, base_weight, _name in pillars:
        # Only the explicit uniform sentinel is an abstention.  A merely
        # low-confidence prediction remains a usable base-weighted vote.
        is_abstaining = np.allclose(probs, uniform, atol=1e-4)
        effective_weights.append(0.0 if is_abstaining else base_weight)

    total_eff_weight = sum(effective_weights)
    if not np.isfinite(total_eff_weight) or total_eff_weight <= 0:
        raise ValueError("Pillar weights must produce a positive finite total.")

    norm_weights = [weight / total_eff_weight for weight in effective_weights]
    final_probs = np.zeros(3, dtype=np.float32)
    for (probs, _, _), weight in zip(pillars, norm_weights):
        final_probs += probs * weight

    # Uneventful Consensus Rule: Widespread abstention means nothing is happening.
    abstain_count = sum(1 for w in effective_weights if w == 0.0)
    if abstain_count >= 4:
        final_probs[1] += 0.30
        final_probs = final_probs / final_probs.sum()

    return final_probs, {
        name: weight
        for (_, _, name), weight in zip(pillars, norm_weights)
    }


def summarize_fusion_result(probs: np.ndarray) -> Tuple[bool, str, int, float]:
    """Return mixed status, display label, runner-up index, and lead margin.

    Mixed is only shown when the runner-up is within MIXED_MARGIN (5 points)
    of the winner. This prevents false Mixed labels when there's a clear winner.
    """
    values = np.asarray(probs, dtype=np.float32)
    if values.shape != (3,):
        raise ValueError("Fusion result must contain Negative, Neutral, and Positive scores.")

    ranked_indices = np.argsort(-values, kind="stable")
    top_index = int(ranked_indices[0])
    runner_up_index = int(ranked_indices[1])
    third_index = int(ranked_indices[2])

    margin = float(values[top_index] - values[runner_up_index])

    top_label = CLASS_NAMES[top_index]
    runner_up_label = CLASS_NAMES[runner_up_index]

    # Mixed only if the runner-up is within MIXED_MARGIN (5 points) of the top
    is_mixed = margin < MIXED_MARGIN

    # Three-way tie: all three within 5 points of each other
    is_three_way_tie = (margin < MIXED_MARGIN and
                        (values[runner_up_index] - values[third_index]) < MIXED_MARGIN)

    if is_three_way_tie:
        display_label = MIXED_ALL_LABEL
    elif is_mixed:
        tie_set = {top_label, runner_up_label}
        if tie_set == {"Positive", "Negative"}:
            # Positive + Negative cancel out = Neutral
            display_label = "Neutral"
        elif tie_set == {"Positive", "Neutral"}:
            display_label = "Positive & Neutral"
        elif tie_set == {"Negative", "Neutral"}:
            display_label = "Negative & Neutral"
        else:
            display_label = f"Mixed: {top_label} / {runner_up_label}"
    else:
        # Clear winner
        display_label = top_label

    return is_mixed, display_label, runner_up_index, margin


def is_mixed_result(probs: np.ndarray) -> bool:
    """Whether the leading score is within five points of the runner-up."""
    return summarize_fusion_result(probs)[0]


# ── UI ────────────────────────────────────────────────────────────────────────
_, col_btn = st.columns([4, 1])
with col_btn:
    if st.button("🗑️ Clear Cache", help="Clears loaded models and memory so you don't have to restart the terminal"):
        with st.spinner("🧹 Deep cleaning all cached resources..."):
            # Clear Streamlit caches
            st.cache_resource.clear()
            st.cache_data.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear PyTorch CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            # Clear any loaded model references
            try:
                import sys
                # Remove model modules from sys.modules to force reload
                modules_to_remove = [k for k in sys.modules.keys() 
                                    if any(x in k for x in ['open_clip', 'whisper', 'transformers', 'clip'])]
                for module in modules_to_remove:
                    del sys.modules[module]
            except Exception:
                pass
            
            # Additional memory cleanup
            gc.collect()
            
            st.success("✨ Cache cleared! All models will reload on next analysis.")
            import time
            time.sleep(1)
        
        st.rerun()

uploaded = st.file_uploader("Drop a video clip (max 15s analyzed)", type=["mp4", "mov", "avi", "mkv"])

if uploaded:
    st.video(uploaded)
    if st.button("✨ Analyze Sentiment", type="primary", use_container_width=True):
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            uploaded.seek(0)
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        wav_path = None
        try:
            with st.spinner("⏳ Extracting video and audio tracks reliably..."):
                frames, wav_path = extract_media(tmp_path)

            colA, colB, colC = st.columns(3)
            with colA:
                with st.spinner("👁️ Spatial (CLIP)..."): clip_probs = run_clip_zeroshot(frames)
                with st.spinner("🎨 Color (HSV)..."): color_probs = run_color(frames)
            with colB:
                with st.spinner("🗣️ Speech (BERT)..."): speech_probs, transcript = run_speech_sentiment(wav_path)
                with st.spinner("🌀 Motion (Flow)..."): motion_probs = run_motion(frames)
            with colC:
                with st.spinner("🎵 Acoustic (Librosa)..."): acous_probs = run_acoustic_emotion(wav_path)
                with st.spinner("⏱️ Temporal (Cuts)..."): temp_probs = run_temporal(frames)

            pillars = [
                (clip_probs, CLIP_W, "Spatial"),
                (speech_probs, SPEECH_W, "Speech"),
                (acous_probs, ACOUS_W, "Acoustic"),
                (color_probs, COLOR_W, "Color"),
                (motion_probs, MOTION_W, "Motion"),
                (temp_probs, TEMP_W, "Temporal")
            ]
            final_probs, effective_weight_by_name = fuse_pillars(pillars)
            pred  = int(final_probs.argmax())
            label = CLASS_NAMES[pred]
            top_score = float(final_probs[pred]) * 100
            mixed_result, mixed_label, runner_up_idx, margin = summarize_fusion_result(final_probs)

            # ── Final result ───────────────────────────────────────────────
            st.markdown("---")
            # The top fusion class is always the model's displayed prediction.
            # A mixed status describes a close runner-up; it never replaces the
            # actual class selected by the fusion model.
            color = COLORS[label]
            emoji = EMOJIS[label]
            lead_detail = (
                "All three fusion scores are within 5.0 points of one another."
                if mixed_label == MIXED_ALL_LABEL else
                f"{label} ({top_score:.1f}%) and {CLASS_NAMES[runner_up_idx]} "
                f"({final_probs[runner_up_idx] * 100:.1f}%) are only {margin * 100:.1f} points apart."
                if mixed_result else ""
            )
            mixed_status_html = (
                f"<p style='text-align:center;color:#f59e0b;'>⚖️ {mixed_label}</p>"
                if mixed_result else ""
            )
            detail_html = (
                f"<p style='text-align:center;color:#64748b;'>{lead_detail}</p>"
                if lead_detail else ""
            )
            st.markdown(
                f"<h2 style='text-align:center;color:{color};'>{emoji} {label}</h2>"
                f"<p style='text-align:center;color:#64748b;'>Top fusion score: {top_score:.1f}%</p>"
                f"{mixed_status_html}{detail_html}",
                unsafe_allow_html=True)

            if transcript and not speech_has_status(transcript):
                st.info(f'**Heard Speech:** "{transcript}"')
            elif "[No speech detected]" in transcript:
                st.caption("*(No speech detected in audio)*")
            elif "[No audio" in transcript:
                st.caption("*(No audio track found in this video)*")
            elif "[Audio too short]" in transcript:
                st.caption("*(Speech signal abstained because the audio was too short)*")
            elif "[Gibberish Detected" in transcript or "[Low Confidence: Muted]" in transcript:
                st.caption("*(Speech signal abstained because the transcript was unreliable)*")
            elif "[Error:" in transcript:
                st.caption("*(Speech signal was unavailable for this clip)*")

            st.markdown("**Fusion scores (not calibrated probabilities):**")
            for i, cls in enumerate(CLASS_NAMES):
                st.progress(float(final_probs[i]), text=f"{EMOJIS[cls]} {cls}: {final_probs[i]*100:.1f}%")

            # ── Per-signal breakdown ───────────────────────────────────────
            st.markdown("---")
            st.markdown("**How each signal voted:**")
            st.caption("Effective weights use the configured base weights for usable signals. Uniform votes abstain and receive 0%.")

            # Row 1: The Heavyweights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"👁️ **Spatial/CLIP** (base {CLIP_W:.0%} · effective {effective_weight_by_name['Spatial']:.0%})")
                for i, cls in enumerate(CLASS_NAMES):
                    st.progress(float(clip_probs[i]), text=f"{cls}: {clip_probs[i]*100:.0f}%")
            with col2:
                st.markdown(f"🗣️ **Audio: Speech** (base {SPEECH_W:.0%} · effective {effective_weight_by_name['Speech']:.0%})")
                for i, cls in enumerate(CLASS_NAMES):
                    st.progress(float(speech_probs[i]), text=f"{cls}: {speech_probs[i]*100:.0f}%")
            with col3:
                st.markdown(f"🎵 **Audio: Acoustic** (base {ACOUS_W:.0%} · effective {effective_weight_by_name['Acoustic']:.0%})")
                for i, cls in enumerate(CLASS_NAMES):
                    st.progress(float(acous_probs[i]), text=f"{cls}: {acous_probs[i]*100:.0f}%")

            st.markdown("---")
            # Row 2: The Structural Pillars
            col4, col5, col6 = st.columns(3)
            with col4:
                st.markdown(f"🎨 **Color Pillar** (base {COLOR_W:.0%} · effective {effective_weight_by_name['Color']:.0%})")
                for i, cls in enumerate(CLASS_NAMES):
                    st.progress(float(color_probs[i]), text=f"{cls}: {color_probs[i]*100:.0f}%")
            with col5:
                st.markdown(f"🌀 **Motion Pillar** (base {MOTION_W:.0%} · effective {effective_weight_by_name['Motion']:.0%})")
                for i, cls in enumerate(CLASS_NAMES):
                    st.progress(float(motion_probs[i]), text=f"{cls}: {motion_probs[i]*100:.0f}%")
            with col6:
                st.markdown(f"⏱️ **Temporal Pillar** (base {TEMP_W:.0%} · effective {effective_weight_by_name['Temporal']:.0%})")
                for i, cls in enumerate(CLASS_NAMES):
                    st.progress(float(temp_probs[i]), text=f"{cls}: {temp_probs[i]*100:.0f}%")

        except Exception as e:
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            try:
                os.unlink(tmp_path)
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception:
                pass
