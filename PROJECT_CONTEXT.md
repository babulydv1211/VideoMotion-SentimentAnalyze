# SceneMotion-LLM — Project Context & Continuation Guide

> Last updated: 2026-09-16 04:18 (PDT)
> Purpose: Load this file at the start of any new chat session to instantly resume work.

---

## 1. What This Project Is

**SceneMotion-LLM** is a video affect (sentiment) classification system.
It classifies short video clips (up to 15 seconds) into **Positive**, **Neutral**, or **Negative** sentiment.

- **Dataset**: LIRIS-ACCEDE — 9,800 movie clips ranked by human emotional valence.
- **Labels** are derived from official `valenceRank` split into fixed global thirds:

| Class    | Rank range  | Class index |
|----------|-------------|-------------|
| Negative | 0 – 3265    | 2           |
| Neutral  | 3266 – 6532 | 1           |
| Positive | 6533 – 9799 | 0           |

- **Official splits**: from `ACCEDEsets.txt` (`1=train`, `2=validation`, `0=test`).
- **UI**: Streamlit app at `scene_motion_llm/app.py`.
- **Package root**: `scene_motion_llm/` (this is where `.git` lives).

---

## 2. Architecture Evolution

### Legacy model (do not use for new work)
- Visual appearance (ResNet) + optical flow via LSTM + temporal attention.
- Found in `scene_motion_llm/models/sentiment_classifier.py`.
- Weakness: learned Neutral-class shortcuts; not truly multimodal.

### Current canonical model: Semantic Five-Pillar Fusion
Five synchronized descriptive feature streams → per-pillar soft P/N/N opinion + reliability score → learned reliability-gated fusion → final label.

```
Video Clip
│
├── Audio Pillar (20 features)
│   ├── RMS energy, dynamic range, zero crossing rate
│   ├── Spectral centroid/rolloff/bandwidth/flatness
│   ├── Onset strength/density, global tempo
│   ├── Low/mid/high band ratio, spectral slope
│   ├── Amplitude modulation, active_audio_fraction
│   └── audio_present flag (gates entire pillar if mute)
│
├── Color Pillar (16 features)
│   ├── brightness, brightness_std, dark/bright pixel fraction
│   ├── saturation, hue (sin+cos), warm/cool balance
│   ├── colorfulness, luminance contrast/entropy
│   └── daylight_proxy, nighttime_proxy
│
├── Spatial Pillar (24 features)
│   ├── edge density/orientation, texture detail
│   ├── luminance entropy, center saliency proxy
│   ├── horizontal/vertical balance, rule-of-thirds proxy
│   ├── face count/area proxy (OpenCV Haar)
│   └── 8× ResNet-18 projection dims (when --use-pretrained-spatial)
│
├── Motion Pillar (14 features)
│   ├── mean/p90/std flow speed, active_motion_fraction
│   ├── global_flow_coherence, direction_entropy
│   ├── global_translation_proxy, motion_acceleration
│   ├── mean horizontal/vertical flow, localized_motion_fraction
│   └── flow_spread, kinematic_intensity_proxy
│
└── Temporal Pillar (12 features)
    ├── relative_time, luminance/color/brightness/saturation delta
    ├── cut_like_change_proxy, motion_speed, motion_acceleration
    ├── visual_activity, cumulative_visual_change
    └── audio_energy_change, audio_visual_sync_proxy
```

**Key design principle**: No pillar hard-codes a sentiment rule.
A dark frame → low `daylight_proxy`, not "Negative". 
A mute clip → `audio_present = 0`, audio fusion weight = exactly 0.
The model *learns* how features relate to valence labels from LIRIS training data.

**Relevant files:**
- `scene_motion_llm/models/semantic_five_pillar.py` — model architecture
- `scene_motion_llm/utils/semantic_pillars.py` — feature extractors
- `scene_motion_llm/training/semantic_five_pillar_training.py` — training pipeline
- `scene_motion_llm/inference/semantic_five_pillar_inference.py` — inference
- `scene_motion_llm/app.py` — Streamlit UI

---

## 3. Current State (as of 2026-09-16)

### ✅ Done
- Complete five-pillar extractor implemented and tested.
- ResNet-18 spatial pretrained embeddings integrated (via `--use-pretrained-spatial`).
- RoBERTa text module scaffolded in `models/audio_modules.py`.
- Streamlit UI updated to show per-pillar evidence, reliability, confidence.
- Unit tests passing: `tests/test_semantic_five_pillar_inference.py` (3/3), `tests/test_semantic_five_pillar_training.py` (3/3).
- `monitor.py` created in project root for live progress monitoring.

### 🔄 In Progress RIGHT NOW
**Feature precomputation is running in background (do not kill).**

```powershell
python -u -m scene_motion_llm.training.semantic_five_pillar_training `
  --mode precompute `
  --video-dir .\scene_motion_llm\dataset\Liris_Accede `
  --ranking-path .\scene_motion_llm\dataset\annotations\ACCEDEranking.txt `
  --sets-path C:\Users\student\Downloads\LIRIS-ACCEDE-annotations\LIRIS-ACCEDE-annotations\annotations\ACCEDEsets.txt `
  --cache-dir .\scene_motion_llm\cache\semantic_five_pillar_v2 `
  --use-pretrained-spatial --spatial-device cpu
```

Monitor progress anytime:
```powershell
python monitor.py
```

Cache target: `scene_motion_llm/cache/semantic_five_pillar_v2/`
- `train/` → ~6,533 clips
- `validation/` → ~1,634 clips
- `test/` → ~1,633 clips

### ❌ Not Started Yet
- PANNs audio tagging integration
- CLIP spatial embeddings
- Full Whisper → RoBERTa wiring into pillar extractor
- Training the model
- Evaluation on test set

---

## 4. Next Steps (Ordered)

### Step 1 — Wait for precompute to finish
Check with `python monitor.py`. When all three splits are full, proceed.

### Step 2 — Run training
```powershell
python -m scene_motion_llm.training.semantic_five_pillar_training `
  --mode train `
  --video-dir .\scene_motion_llm\dataset\Liris_Accede `
  --ranking-path .\scene_motion_llm\dataset\annotations\ACCEDEranking.txt `
  --sets-path C:\Users\student\Downloads\LIRIS-ACCEDE-annotations\LIRIS-ACCEDE-annotations\annotations\ACCEDEsets.txt `
  --cache-dir .\scene_motion_llm\cache\semantic_five_pillar_v2 `
  --checkpoint-dir .\scene_motion_llm\checkpoints\semantic_five_pillar `
  --epochs 25 --batch-size 16 --device cuda --use-pretrained-spatial
```
(Use `--device cpu` if no GPU available — will be slower.)

Best checkpoint selected by validation macro-F1 → saved as `best_semantic_five_pillar.pt`.

### Step 3 — Establish baseline accuracy
Expected honest range: **~55–65% balanced accuracy**, **macro-F1 ~0.55–0.60**.
This is the baseline. Any number below 50% means something is wrong (worse than random for 3 balanced classes is a red flag).

### Step 4 — Upgrade pillars with free local pretrained models

All models below are **free, local, no API key, reproducible**:

| Pillar | Upgrade | Model | License | Notes |
|--------|---------|-------|---------|-------|
| Audio | Audio event tagging (music, speech, alarm, ambience) | **PANNs CNN14** | Apache 2.0 ✅ | ~300MB, download once |
| Audio | Speech transcription | **Whisper tiny/base** | MIT ✅ | Already in requirements.txt |
| Audio | Transcript → sentiment embedding | **RoBERTa-base** | MIT ✅ | Already scaffolded in audio_modules.py |
| Spatial | Scene/object/concept embeddings | **CLIP ViT-B/32** (OpenAI) | MIT ✅ | Replaces ResNet, much richer |
| Motion | Better optical flow | **RAFT-small** | BSD ✅ | ~5MB, very fast |

**All models are free forever, run 100% locally, no API key, no quota, no billing.**

**Explicitly excluded (do not add):**
- ~~YOLOv8~~ → AGPL-3.0, requires paid enterprise license for any non-open product
- ~~Hugging Face Inference API~~ → only $0.10/month free, then pay-as-you-go, non-reproducible
- ~~Any cloud vision/NLP API~~ → Google Vision, AWS Rekognition, Azure CV all have per-request costs

### Step 5 — Retrain with enhanced pillars and measure delta
Compare baseline F1 vs enhanced F1. Expected uplift from CLIP + PANNs: **+5 to +15 percentage points**.

---

## 5. Architectural Philosophy (Important — Do Not Break These Rules)

1. **Sub-features are evidence, not rules.**
   `dark_pixel_fraction = 0.8` → "evidence leaning Negative" not "= Negative".
   A dark romantic scene with happy audio might still be Positive. The fusion gate decides.

2. **Missing modalities are first-class.**
   Mute video: audio pillar weight = 0 automatically.
   Black/low-detail frame: spatial/color reliability scores drop but pillars stay enabled.

3. **No cloud APIs in the training or inference pipeline.**
   Download weights once, cache locally, train and infer offline forever.

4. **Per-pillar soft P/N/N opinions feed into fusion.**
   Each pillar outputs a (3,) soft probability vector + scalar reliability.
   Fusion gate learns to weight and combine these per-clip.

5. **Ordinal loss is worth trying.**
   Negative→Neutral→Positive has order. Ordinal cross-entropy penalizes
   predicting Positive for Negative more than predicting Neutral.
   Add this once baseline is established.

---

## 6. Repo Layout

```
sentiment_analysis/              ← workspace root (no .git here)
├── monitor.py                   ← live precompute progress tracker
├── PROJECT_CONTEXT.md           ← THIS FILE
├── setup.py                     ← installs scene_motion_llm package
├── requirements.txt             ← points to scene_motion_llm/requirements.txt
├── tests/                       ← pytest test suite
│   ├── test_semantic_five_pillar_inference.py   ✅ 3 pass
│   ├── test_semantic_five_pillar_training.py    ✅ 3 pass
│   └── test_inference_checkpoint.py             ⚠️ needs transformers in venv
└── scene_motion_llm/            ← actual package (.git lives here)
    ├── app.py                   ← Streamlit UI
    ├── train_pipeline.py        ← CLI entry point
    ├── models/
    │   ├── semantic_five_pillar.py   ← CANONICAL MODEL
    │   ├── sentiment_classifier.py  ← legacy (keep but don't train)
    │   └── audio_modules.py         ← AudioLSTM + TextRoBERTa
    ├── inference/
    │   └── semantic_five_pillar_inference.py
    ├── training/
    │   └── semantic_five_pillar_training.py
    ├── utils/
    │   └── semantic_pillars.py       ← ALL FEATURE EXTRACTORS
    ├── cache/
    │   └── semantic_five_pillar_v2/  ← precomputed features (in progress)
    └── checkpoints/
        └── semantic_five_pillar/     ← training will write here
```

---

## 7. How to Run the Streamlit App

```powershell
# From project root (sentiment_analysis/)
.\.venv\Scripts\Activate.ps1
streamlit run scene_motion_llm\app.py
```

The app will:
- Auto-detect the best available checkpoint
- Show "Five-pillar checkpoint found" if `best_semantic_five_pillar.pt` exists
- Fall back to legacy model with a warning if only old checkpoint found
- Show per-pillar evidence bars + confidence after upload

---

## 8. Key Design Decisions Made

| Decision | Reason |
|----------|--------|
| Use LIRIS official splits (ACCEDEsets.txt) | Avoids random-split data leakage; makes results reproducible |
| Fixed global thirds for labels | Avoids per-split threshold drift; consistent across runs |
| Reliability gating over hard missing-modality rules | Graceful degradation for any real-world clip |
| Cache features before training | ~9,800 × 5 pillars is expensive; caching lets you iterate on the model fast |
| ResNet-18 over larger models for spatial | Speed / memory tradeoff for 9,800-video cache on CPU |
| CLIP over YOLO for spatial upgrade | CLIP is MIT licensed and gives semantic scene embeddings, not just boxes |

---

## 9. Open Questions / Things to Decide Later

- [ ] Should we add ordinal loss to the training objective?
- [ ] Should we try continuous valence regression as an auxiliary loss?
- [ ] Should we create a small real-world test set (100–300 clips, 3 human annotators)?
- [ ] Should CLIP replace ResNet or be added as extra dimensions in the spatial pillar?
- [ ] Should PANNs audio tags be concatenated to acoustic features or be a separate sub-model?
- [ ] Should Whisper run on every clip or only when `audio_present > 0.5`?

---

## 10. How to Resume in a New Chat

Paste this at the start of a new conversation:
```
I'm continuing work on SceneMotion-LLM video sentiment classification.
Please read PROJECT_CONTEXT.md at the project root 
(c:\Users\student\Desktop\sentiment_analysis\PROJECT_CONTEXT.md)
and then tell me the current state and what to do next.
```

The agent will read this file and have full context instantly.
