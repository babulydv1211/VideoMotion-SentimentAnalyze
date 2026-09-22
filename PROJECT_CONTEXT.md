# SceneMotion AI — Project Context & Quick Start

> Last updated: 2026-09-21
> Purpose: Quick onboarding guide for new developers or continuation after a break

---

## What This Project Is

**SceneMotion AI** is a **zero-shot, rule-based video sentiment classifier** that analyzes short clips (up to 15 seconds) and outputs **Negative**, **Neutral**, or **Positive** sentiment.

**Key features:**
- ✅ Works immediately (no training required)
- ✅ Local inference after the required model weights have been downloaded once
- ✅ Fully explainable (shows per-pillar voting breakdown)
- ✅ Transparent, hand-authored rules and per-pillar vote breakdowns

---

## How It Works (High-Level)

The system uses **6 independent expert modules** (pillars) that each analyze a different aspect of the video:

```
Video → MoviePy extraction → 16 frames + audio
    ↓
┌─────────────────────────────────────────────────┐
│ P1: CLIP (40%)        → Visual semantics        │
│ P2: Whisper+BERT (10%)→ Spoken words            │
│ P3: Librosa (25%)     → Audio tone (waveform)   │
│ P4: OpenCV HSV (5%)   → Color mood              │
│ P5: Optical Flow (15%)→ Motion intensity        │
│ P6: Frame Diff (5%)   → Shot cuts               │
└─────────────────────────────────────────────────┘
    ↓
Dynamic entropy-weighted fusion
    ↓
Final: Negative / Neutral / Positive (+ confidence)
```

**Dynamic fusion:** Each pillar is downweighted according to the entropy of its own score distribution. A uniform non-speech vote retains 20% of its base weight before the weights are normalized; a uniform speech vote abstains entirely. This is a heuristic reliability rule, not calibrated uncertainty.

---

## Project Structure

```
sentiment_analysis/               ← workspace root
├── app.py                        ← MAIN ENTRY POINT (rule-based system)
├── README.md                     ← Full technical documentation
├── PROJECT_CONTEXT.md            ← This file
├── requirements.txt              ← Python dependencies
├── .codex-venv/                  ← Local virtual environment (ignored by Git)
├── utils/                        ← Utility functions (if any)
├── tests/                        ← Unit tests for active rules and shared pipeline utilities
└── archive/                      ← Neural training system (archived)
    ├── training/                 ← LIRIS-ACCEDE training pipeline
    ├── models/                   ← Sequence fusion model
    ├── inference/                ← Checkpoint-based inference
    ├── app.py                    ← Old neural Streamlit UI
    └── test_*.py                 ← Neural model tests
```

---

## Quick Start

### 1. Activate Virtual Environment

```powershell
# Windows PowerShell
.\.codex-venv\Scripts\Activate.ps1
```

### 2. Run the Streamlit App

```powershell
python -m streamlit run app.py
```

### 3. Use the System

1. Open browser (Streamlit auto-launches to `localhost:8501`)
2. Upload a video file (max 15 seconds, any common format)
3. Click "Analyze Sentiment"
4. View results:
   - Final sentiment (Neg/Neu/Pos) with confidence %
   - Per-pillar breakdown (how each module voted)
   - Transcript (if speech detected)
   - Evidence details (motion, color, etc.)

---

## Dependencies

**Core models (downloaded once, cached locally):**
- **CLIP** `ViT-B-32` — Visual semantics (~350MB)
- **Whisper** `base` — Speech-to-text (~140MB)
- **DistilBERT** emotion — Text sentiment (~260MB)
- **Librosa** — Audio feature extraction (DSP, no model)
- **OpenCV** — Computer vision primitives

**Python packages:**
- `streamlit` — Web UI
- `open_clip_torch` — CLIP interface
- `openai-whisper` — Speech recognition
- `transformers` — DistilBERT
- `librosa` — Audio analysis
- `moviepy` — Video I/O
- `opencv-python` — Vision processing
- `torch` — Backend for neural models
- `numpy`, `Pillow` — Standard numerical/image tools

All dependencies are in `requirements.txt`.

---

## Key Design Decisions

### Why Rule-Based Instead of Learned?

1. **No training data needed** — runs once its pretrained model weights are available locally
2. **Inspectable** — per-pillar scores and effective fusion weights can be reviewed
3. **Deterministic given fixed model/runtime versions** — model upgrades and video decoding can still affect results
4. **Transparent heuristic defaults** — weights are documented rather than learned

### Why These Specific Weights?

| Pillar | Weight | Justification |
|---|---|---|
| Spatial (CLIP) | 40% | Primary visual-semantic signal |
| Acoustic | 25% | Primary audio-prosody signal |
| Motion | 15% | Auxiliary motion/arousal cue |
| Speech | 10% | Transcript/emotion cue with abstention safeguards |
| Color | 5% | Mood + arousal secondary signal |
| Temporal | 5% | Shot cut tension indicator |

**Dynamic adjustment:** each pillar's base weight is multiplied by a score-concentration factor derived from Shannon entropy, then all effective weights are normalized. Uniform speech is an abstention. These are heuristic weights, not calibrated probabilities or a validated psychological measurement scale.

### Why Not Train on LIRIS-ACCEDE?

An earlier experiment reported about **33% accuracy** while tuning on LIRIS-ACCEDE, whose labels describe human viewer affect. That result should be treated as a warning that this rule-based classifier has not been validated for that target, not as proof that the mismatch is the only cause.

**Root cause discovered:** LIRIS labels measure **induced emotion** (how viewers *feel*), while our system measures **expressed emotion** (what signals *exist* in the video). These are fundamentally different:
- A horror movie → viewer feels fear (induced: Negative) but the video itself may show calm imagery before the jumpscare (expressed: Neutral)
- A sad documentary → viewer cries (induced: Negative) but narrator speaks calmly (expressed: partial Neutral)

Training on ACCEDE may optimize for viewer-induced affect rather than the content cues this demo uses. Choose and document a labeled evaluation set that matches the intended task before making performance claims.

---

## Archived Neural System

The `archive/` directory contains a **trainable sequence fusion model** that was developed but ultimately not used. It includes:
- Full LIRIS-ACCEDE training pipeline
- Feature extraction + caching system
- BiGRU temporal sequence encoder
- Learned reliability-weighted fusion

**Why archived:**
- Requires 9,800-video dataset + hours of preprocessing
- Learned weights aren't more explainable
- Empirical tuning hit the induced/expressed emotion gap
- Rule-based system already works well

The archived code is fully functional if you want to experiment with learned fusion in the future.

---

## Testing

Run tests to verify the system:

```powershell
# Run all tests with the project environment
.\.codex-venv\Scripts\python.exe -m pytest -q

# Run focused active-app rule tests
.\.codex-venv\Scripts\python.exe -m pytest tests/test_rule_based_pillars.py -q
```

**Note:** Neural model tests have been moved to `archive/`. Active tests focus on the rule-based system.

---

## Common Issues

### "Model not found" errors on first run
**Solution:** Models download automatically on first use. Ensure internet connection for initial setup (~750MB total).

### Out of memory (OOM)
**Solution:**
- Use CPU mode: Set `DEVICE = "cpu"` in `app.py`
- Reduce batch size if processing multiple videos
- Click "Clear Cache" in the app header between analyses

### Whisper fails / no transcript
**Solution:** Whisper requires `ffmpeg`. Install via:
```powershell
# Windows (requires Chocolatey)
choco install ffmpeg
```

### Video upload fails
**Solution:** MoviePy uses `ffmpeg` for decoding. Supported upload formats are MP4, AVI, MOV, and MKV. The app analyzes the first 15 seconds of longer clips.

---

## Next Steps / Future Work

**Potential enhancements:**
1. **Batch processing** — analyze multiple videos in one session
2. **Export results** — save analysis as JSON/CSV
3. **Threshold tuning** — let users adjust pillar weights via UI sliders
4. **Custom prompts** — allow users to define their own CLIP text prompts
5. **Face detection integration** — boost spatial confidence when faces detected
6. **Audio event tagging** — add PANNs for music/speech/ambient classification

**Not recommended:**
- Training on LIRIS-ACCEDE (induced vs expressed emotion mismatch)
- Adding more pillars (diminishing returns, complexity increases)
- Cloud API integration (defeats the local/offline value proposition)

---

## References

- **Mehrabian & Ferris (1967).** Inference of attitudes from nonverbal communication in two channels. *Journal of Consulting Psychology*.
- **Russell (1980).** A circumplex model of affect. *Journal of Personality and Social Psychology*.
- **Ekman (1992).** An argument for basic emotions. *Cognition & Emotion*.
- **Radford et al. (2021).** Learning Transferable Visual Models From Natural Language Supervision. *ICML*. (CLIP)
- **Cutting et al. (2010).** Attention and the evolution of Hollywood film. *Psychological Science*.

---

## Questions?

Read `README.md` for full technical details on each pillar, fusion algorithm, and experimental validation.
