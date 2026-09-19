# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — Semantic Five-Pillar v2 (in progress)

### In Progress
- Feature precomputation of all 9,800 LIRIS-ACCEDE clips into `cache/semantic_five_pillar_v2/`
- Model training not yet started (runs after cache completes via `run_train.ps1`)

### Planned
- PANNs CNN14 audio tagging (Apache 2.0) inside audio pillar
- CLIP ViT-B/32 (MIT) scene embeddings inside spatial pillar
- Whisper → RoBERTa transcript sentiment wired into audio pillar

---

## [2.0.0] — Semantic Five-Pillar Architecture

### Added
- `models/semantic_five_pillar.py` — reliability-gated five-pillar fusion model
- `utils/semantic_pillars.py` — five synchronized feature extractors (audio, color, spatial, motion, temporal)
- `training/semantic_five_pillar_training.py` — canonical LIRIS-ACCEDE training using official splits
- `inference/semantic_five_pillar_inference.py` — upload-time inference for the new model
- `models/audio_modules.py` — AudioLSTM and TextRoBERTa scaffolding for multimodal audio+text
- `conftest.py` — pytest path fix (no more PYTHONPATH=. required)
- `monitor.py` — live terminal dashboard for precompute progress
- `run_train.ps1` — one-click training launcher
- `evaluate.py` — post-training metrics printer (accuracy, F1, confusion matrix)
- `PROJECT_CONTEXT.md` — full project context for resuming across sessions

### Changed
- `app.py` — updated Streamlit UI to support five-pillar backend, per-pillar evidence display, audio status, legacy fallback
- `inference/__init__.py` — lazy-load legacy model; canonical path is now semantic five-pillar
- `requirements.txt` — added `imageio-ffmpeg==0.5.1` for reliable audio decoding

### Design Decisions
- Labels from LIRIS official `ACCEDEsets.txt` splits (not random splits)
- Fixed global thirds of `valenceRank` for Negative/Neutral/Positive
- Reliability gating: mute clips get audio weight = 0, not a fallback label
- All pretrained models run locally (MIT/Apache 2.0/BSD only — no AGPL, no cloud APIs)

---

## [1.0.0] — Legacy Visual/Motion Model

### Added
- Original ResNet + optical flow LSTM + temporal attention classifier
- Streamlit UI (basic)
- LIRIS-ACCEDE training with random splits (deprecated — use official splits)
- Stage1/Stage2 checkpoint training (UCF101 → LIRIS transfer)

### Known Issues (reason for v2 rewrite)
- Strong Neutral-class shortcut (~64.8% accuracy from majority-class bias)
- Random splits caused data leakage between runs
- No audio or text modality
- Hard-coded visual rules instead of learned reliability gates
