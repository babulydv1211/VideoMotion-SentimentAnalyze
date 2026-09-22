"""
Weight Optimizer — Loads models ONCE, then batches all 60 videos fast.
"""
import sys, os, warnings, tempfile
warnings.filterwarnings("ignore")
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
sys.path.append(r"C:\Users\student\Desktop\sentiment_analysis")

import numpy as np
import pandas as pd
from itertools import product

# ── Ground truth ─────────────────────────────────────────────────────────────
GT_FILE   = r"C:\Users\student\Desktop\sentiment_analysis\scene_motion_llm\dataset\annotations\ACCEDEranking.txt"
VIDEO_DIR = r"C:\Users\student\Desktop\sentiment_analysis\scene_motion_llm\dataset\Liris_Accede"

df = pd.read_csv(GT_FILE, sep="\t")
def valence_to_label(v):
    if v < 2.5:   return 0  # Negative
    elif v > 3.5: return 2  # Positive
    else:         return 1  # Neutral
df["label"] = df["valenceValue"].apply(valence_to_label)
sample = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(20, len(g)), random_state=42)).reset_index(drop=True)
print(f"Using {len(sample)} balanced samples: {sample['label'].value_counts().to_dict()}")

# ── Load all models ONCE ──────────────────────────────────────────────────────
print("\nLoading models (once)...")

import torch, cv2, librosa
import open_clip
from PIL import Image
from moviepy import VideoFileClip
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DURATION = 15.0
SAMPLED_FRAMES = 16

print("  [1/3] Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.eval().to(DEVICE)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

TEXT_PROMPTS = {
    "Negative": ["a video scene conveying sadness","a video scene conveying fear","a video scene conveying anger","a video scene conveying disgust"],
    "Neutral":  ["a video scene conveying calm","a video scene conveying surprise","a neutral everyday scene"],
    "Positive": ["a video scene conveying joy","a video scene conveying love","a video scene conveying happiness"],
}
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
all_texts = [p for cls in CLASS_NAMES for p in TEXT_PROMPTS[cls]]
text_tokens = tokenizer(all_texts).to(DEVICE)
with torch.no_grad():
    text_feats = clip_model.encode_text(text_tokens)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)

print("  [2/3] Loading Whisper (STT)...")
import whisper
stt_model = whisper.load_model("base", device=DEVICE)

print("  [3/3] Loading DistilBERT (Emotion)...")
emotion_pipe = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None, device=0 if DEVICE=="cuda" else -1)

print("Models loaded!\n")

# ── Helper functions ──────────────────────────────────────────────────────────
def extract_media(path):
    clip = VideoFileClip(path)
    dur = min(clip.duration, MAX_DURATION)
    fps = clip.fps if clip.fps and clip.fps > 0 else 25.0
    total_frames = int(fps * dur)
    indices = np.linspace(0, max(total_frames-1, 0), SAMPLED_FRAMES, dtype=int)
    frames = [clip.get_frame(idx/fps) for idx in indices]
    wav_path = path + "_audio.wav"
    if clip.audio is not None:
        clip.audio.write_audiofile(wav_path, fps=22050, logger=None)
    else:
        wav_path = None
    clip.close()
    return frames, wav_path

def run_clip(frames):
    imgs = []
    for f in frames:
        img = Image.fromarray(f.astype(np.uint8))
        imgs.append(preprocess(img))
    img_tensor = torch.stack(imgs).to(DEVICE)
    with torch.no_grad():
        img_feats = clip_model.encode_image(img_tensor)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
    sims = (img_feats @ text_feats.T).mean(0).softmax(-1).cpu().numpy()
    probs_per_class = np.zeros(3, dtype=np.float32)
    idx = 0
    for ci, cls in enumerate(CLASS_NAMES):
        n = len(TEXT_PROMPTS[cls])
        probs_per_class[ci] = sims[idx:idx+n].mean()
        idx += n
    return probs_per_class / probs_per_class.sum()

def run_acoustic(wav_path):
    if not wav_path or not os.path.exists(wav_path):
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)
        
    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True)
        if len(y) < sr * 0.5:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)

        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_mean, rms_std = float(np.mean(rms)), float(np.std(rms))
        
        if rms_mean < 1e-3:
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

        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
        zcr_std = float(np.std(zcr))

        spec = np.abs(librosa.stft(y, hop_length=512))
        flux = float(np.clip(np.log1p(np.mean(np.diff(spec, axis=1) ** 2)) / 10.0, 0, 1))

        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rhythmic = float(np.clip(float(tempo) / 180.0, 0, 1)) if not np.isnan(float(tempo)) else 0.3
        except Exception:
            rhythmic = 0.3

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bright = float(np.clip(np.mean(centroid) / 4000.0, 0, 1))

        # Harmonicity check (Music vs. Noise)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic**2) + 1e-6
        percussive_energy = np.sum(y_percussive**2) + 1e-6
        is_musical = harmonic_energy > (percussive_energy * 1.5)

        distress = np.clip(0.35 * pitch_cv * 2.0 + 0.25 * rms_cv + 0.20 * flux + 0.20 * pitch_high, 0, 1)
        if is_musical:
            distress = distress * 0.3 # Suppress distress rule if it is harmonic music

        energy_pos = np.clip(0.40 * rhythmic + 0.30 * bright + 0.30 * rms_mean * 10.0, 0, 1)
        neutral = float(np.clip(1.0 - distress - energy_pos * 0.5, 0, 1))

        raw = np.array([distress, neutral, energy_pos], dtype=np.float32)
        raw = np.clip(raw, 1e-3, None)
        return raw / raw.sum()

    except Exception as e:
        print(f"Error in Acoustic: {e}")
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)

def run_speech(wav_path):
    if not wav_path or not os.path.exists(wav_path):
        return np.array([1/3,1/3,1/3], dtype=np.float32)
    try:
        y, _ = librosa.load(wav_path, sr=16000, mono=True)
        if len(y) < 16000 * 0.5:
            return np.array([1/3,1/3,1/3], dtype=np.float32)
        result = stt_model.transcribe(wav_path, fp16=False)
        text = result["text"].lower().strip()
        if not text:
            return np.array([1/3,1/3,1/3], dtype=np.float32)

        # Lexical Validity Filter
        from spellchecker import SpellChecker
        import re
        spell = SpellChecker()
        words = re.findall(r'\b[a-z]+\b', text)
        if len(words) > 0:
            known_words = spell.known(words)
            if len(known_words) / len(words) < 0.60:
                return np.array([1/3,1/3,1/3], dtype=np.float32)
        scores = emotion_pipe(text)[0]
        neg, neu, pos = 0.0, 0.0, 0.0
        for s in scores:
            if s["label"] == "negative": neg += s["score"]
            elif s["label"] == "positive": pos += s["score"]
            elif s["label"] == "neutral": neu += s["score"]
        if max(neg, pos, neu) < 0.50:
            return np.array([1/3,1/3,1/3], dtype=np.float32)
        raw = np.array([neg, neu, pos], dtype=np.float32)
        raw = np.clip(raw, 1e-3, None)
        return raw / raw.sum()
    except:
        return np.array([1/3,1/3,1/3], dtype=np.float32)

def run_color(frames, valence_probs):
    if not frames: return np.array([1/3,1/3,1/3], dtype=np.float32)
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
            
        dark = 1.0 - (np.mean(v) / 255.0)
        darkness_scores.append(dark)
        
        bright = np.mean(v) / 255.0
        brightness_scores.append(bright)
        
        red_mask = ((h < 10) | (h > 170)) & (s > 100) & (v > 50)
        redness_scores.append(np.sum(red_mask) / (h.size + 1e-6))

        green_mask = (h > 35) & (h < 85) & (s > 50) & (v > 50)
        greenness_scores.append(np.sum(green_mask) / (h.size + 1e-6))

        blue_mask = (h > 100) & (h < 140) & (s > 50) & (v > 50)
        blueness_scores.append(np.sum(blue_mask) / (h.size + 1e-6))

        yellow_mask = (h >= 20) & (h <= 35) & (s > 50) & (v > 50)
        yellowness_scores.append(np.sum(yellow_mask) / (h.size + 1e-6))

    avg_sat = np.mean(saturations)
    is_bw = avg_sat < 15
    hue_std = np.std(valid_hues) if len(valid_hues) > 100 else 0
    is_filtered = hue_std < 25 and not is_bw
    
    if is_bw or is_filtered:
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    avg_dark = float(np.mean(darkness_scores))
    avg_bright = float(np.mean(brightness_scores))
    avg_red = float(np.mean(redness_scores)) * 10.0
    avg_green = float(np.mean(greenness_scores)) * 10.0
    avg_blue = float(np.mean(blueness_scores)) * 10.0
    avg_yellow = float(np.mean(yellowness_scores)) * 10.0

    base_neg = np.clip(avg_dark * 0.7, 0, 1)
    base_pos = np.clip(avg_bright, 0, 1)
    base_neu = float(np.clip(1.0 - base_neg - base_pos, 0, 1))

    clip_neg, clip_neu, clip_pos = valence_probs
    if avg_red > 0.1:
        if clip_neg > clip_pos:
            base_neg = np.clip(base_neg + avg_red, 0, 1)
            base_neu = base_neu * 0.5
        elif clip_pos > clip_neg:
            base_pos = np.clip(base_pos + avg_red, 0, 1)
            base_neu = base_neu * 0.5

    if avg_green > 0.1:
        base_neg = np.clip(base_neg + avg_green, 0, 1)
        
    if avg_yellow > 0.1:
        base_pos = np.clip(base_pos + avg_yellow, 0, 1)

    if avg_blue > 0.2:
        suppression = np.clip(avg_blue, 0, 1)
        base_neg = base_neg * (1.0 - suppression)
        base_pos = base_pos * (1.0 - suppression)
        base_neu = base_neu + suppression

    raw = np.array([base_neg, base_neu, base_pos], dtype=np.float32)
    raw = np.clip(raw, 1e-3, None)
    return raw / raw.sum()

def run_motion(frames, valence_probs):
    if len(frames) < 2: return np.array([1/3,1/3,1/3], dtype=np.float32)
    flow_mags = []
    angles = []
    accelerations = []
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    prev_mag = None
    
    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        flow_mags.append(np.std(mag))
        
        moving_pixels = mag > 1.0
        if np.any(moving_pixels):
            angles.append(np.std(ang[moving_pixels]))
        else:
            angles.append(0)
            
        if prev_mag is not None:
            accel = np.mean(np.abs(mag - prev_mag))
            accelerations.append(accel)
            
        prev_mag = mag
        prev_gray = gray

    avg_flow_std = float(np.mean(flow_mags))
    chaos = float(np.mean(angles))
    erratic = float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0
    
    arousal = np.clip(avg_flow_std / 3.0, 0, 1) 
    
    neg, pos = 0.0, 0.0
    if chaos > 0.8:
        neg += np.clip((chaos - 0.8) * 0.5, 0, 0.4)
    if erratic > 3.0:
        neg += np.clip((erratic - 3.0) * 0.1, 0, 0.4)
    if arousal > 0.3 and chaos < 0.6 and erratic < 2.0:
        pos += np.clip(arousal * 0.5, 0, 0.4)

    active_neg = valence_probs[0] + neg
    active_pos = valence_probs[2] + pos
    active_neu = 0.1
    active_vote = np.array([active_neg, active_neu, active_pos], dtype=np.float32)
    active_vote /= active_vote.sum()
    
    baseline = np.array([1/3,1/3,1/3], dtype=np.float32)
    out = (1-arousal)*baseline + arousal*active_vote
    return (out/out.sum()).astype(np.float32)

def run_temporal(frames, valence_probs):
    if len(frames) < 2: return np.array([1/3,1/3,1/3], dtype=np.float32)
    hist_corrs = []
    dutch_frames = 0
    brightness_vals = []
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
    cv2.normalize(prev_hist, prev_hist)
    
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        brightness_vals.append(np.mean(cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[..., 2]))
        
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                angle = angle if angle <= 90 else 180 - angle
                angles.append(angle)
            dutch_lines = sum(1 for a in angles if (10 < a < 80))
            if len(angles) > 0 and (dutch_lines / len(angles)) > 0.4:
                dutch_frames += 1

    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        curr_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(curr_hist, curr_hist)
        corr = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        hist_corrs.append(corr)
        prev_hist = curr_hist
        
    neg, pos = 0.0, 0.0
    if brightness_vals[-1] < 20 and brightness_vals[0] > 50:
        neg += 0.4
    dutch_ratio = dutch_frames / len(frames)
    if dutch_ratio > 0.3:
        neg += np.clip(dutch_ratio, 0, 0.5)
    cuts = sum(1 for c in hist_corrs if c < 0.5)
    if cuts >= 2:
        neg += np.clip(cuts / 5.0, 0, 0.4)
    if cuts == 0 and dutch_ratio < 0.1:
        pos += 0.3

    neg = np.clip(neg, 0, 1)
    pos = np.clip(pos, 0, 1)
    neu = np.clip(1.0 - neg - pos, 0, 1)
    
    out = np.array([neg, neu, pos], dtype=np.float32)
    out = np.clip(out, 1e-3, None)
    return (out/out.sum()).astype(np.float32)

# ── Collect pillar votes for all 60 videos ────────────────────────────────────
results = []
for i, row in sample.iterrows():
    path = os.path.join(VIDEO_DIR, row["name"])
    if not os.path.exists(path):
        continue
    print(f"[{i+1}/{len(sample)}] {row['name']} GT={['Neg','Neu','Pos'][row['label']]}", end=" ", flush=True)
    try:
        frames, wav = extract_media(path)
        clip_p   = run_clip(frames)
        acous_p  = run_acoustic(wav)
        speech_p = run_speech(wav)
        color_p  = run_color(frames)
        motion_p = run_motion(frames)
        temp_p   = run_temporal(frames)
        
        results.append({
            "label": int(row["label"]), 
            "clip_p": clip_p,
            "speech_p": speech_p,
            "acous_p": acous_p,
            "color_p": color_p,
            "motion_p": motion_p,
            "temp_p": temp_p
        })
        print(f"clip={['N','U','P'][int(clip_p.argmax())]} acous={['N','U','P'][int(acous_p.argmax())]} OK", flush=True)
        if wav and os.path.exists(wav): os.unlink(wav)
    except Exception as e:
        print(f"ERR: {e}", flush=True)

print(f"\n[DONE] {len(results)}/{len(sample)} videos processed. Running grid search...\n")

# ── Grid search ───────────────────────────────────────────────────────────────
cands = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
best_acc, best_w = 0, None
tested = 0
for clip_w, speech_w, acous_w, motion_w in product(cands, repeat=4):
    if abs(clip_w+speech_w+acous_w+motion_w+0.10 - 1.0) > 0.01: continue
    if clip_w < 0.20: continue
    if acous_w <= speech_w: continue
    tested += 1
    correct = 0
    for r in results:
        weights = [clip_w, speech_w, acous_w, 0.05, motion_w, 0.05]
        probs_list = [r["clip_p"], r["speech_p"], r["acous_p"], r["color_p"], r["motion_p"], r["temp_p"]]
        
        effective_weights = []
        uniform = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        for p, w in zip(probs_list, weights):
            is_abstaining = np.allclose(p, uniform, atol=1e-4)
            effective_weights.append(0.0 if is_abstaining else w)
            
        total_eff_weight = sum(effective_weights)
        if total_eff_weight <= 0:
            total_eff_weight = 1.0
            
        norm_weights = [w / total_eff_weight for w in effective_weights]
        
        final = np.zeros(3, dtype=np.float32)
        for p, w in zip(probs_list, norm_weights):
            final += p * w
            
        abstain_count = sum(1 for w in effective_weights if w == 0.0)
        if abstain_count >= 4:
            final[1] += 0.30
            final = final / final.sum()
                 
        if int(final.argmax()) == r["label"]:
            correct += 1
            
    acc = correct/len(results)
    if acc > best_acc:
        best_acc, best_w = acc, (clip_w, speech_w, acous_w, motion_w)

print(f"Tested {tested} combinations.\n")
print(f"[BEST] BEST WEIGHTS (accuracy={best_acc*100:.1f}% on {len(results)} videos):")
print(f"  CLIP_W    = {best_w[0]}")
print(f"  SPEECH_W  = {best_w[1]}")
print(f"  ACOUS_W   = {best_w[2]}")
print(f"  MOTION_W  = {best_w[3]}")
print(f"  COLOR_W   = 0.05")
print(f"  TEMP_W    = 0.05")
print(f"  SUM       = {sum(best_w)+0.10:.2f}")
