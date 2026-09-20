# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — Overnight Training (queued)

### Next
- Full 40-epoch overnight training of 5-pillar neural model with CLIP spatial pillar
- Target: 50–62% test macro-F1 (SOTA ceiling ~65%)
- Command: `semantic_five_pillar_training --epochs 40 --embedding-dim 128 ...` (see README)

---

## [3.0.0] — CLIP ViT-B/32 Spatial Pillar Integration  *(2026-09-19)*

### Experiment Results (cumulative)

| Stage | Model | Test Accuracy | Test Macro-F1 |
|-------|-------|:---:|:---:|
| Baseline (v1) | ResNet18 + random split | ~64.8% | — (majority-class shortcut) |
| Rewrite (v2) | ResNet18 + MFCC + Optical Flow, official splits | ~40% | ~40% |
| CLIP MLP (v3a) | CLIP ViT-B/32 + MLP, official splits | 43.82% | 43.95% |
| **5-Pillar Neural (v3b)** | **CLIP + Audio + Color + Motion + Temporal, BiGRU fusion** | **TBD** | **TBD** |

> The v1 "64.8% accuracy" was a Neutral-class shortcut (majority-class bias), not real learning.
> The dataset's human-labelling noise caps genuine models at ~60–65%.

### Added
- `clip_precompute.py` — extracts 512-dim L2-normalized CLIP ViT-B/32 embeddings for all
  9,800 LIRIS-ACCEDE clips (mean of 16 frames). Cached to `cache/clip_vitb32_v1/`.
- `clip_fusion_train.py` — sklearn MLP baseline fusing CLIP visual + MFCC audio + optical
  flow motion features. Grid-searches hidden layer size. Reports val + test metrics.
- `merge_clip_cache.py` — one-off data migration script that swaps the 24-dim handcrafted
  spatial features for 512-dim CLIP embeddings in the canonical five-pillar cache.
  Produces `cache/semantic_five_pillar_v3_clip/` (9,800 merged `.npz` files).
- `README.md` (root) — GitHub landing page with architecture overview, SOTA context,
  inference instructions, repo structure, and web UI guide.

### Changed
- `training/semantic_five_pillar_training.py`:
  - Spatial pillar input dimension upgraded from **24 → 512** to accept CLIP features.
  - `_cache_is_current()` bypass added to accept the custom v3 merged cache.
  - `pillar_dims["spatial"] = 512` injected in both `train` mode code paths.
- `scene_motion_llm/README.md` — added **State-of-the-Art & Benchmark Context** section
  explaining the 60–65% ceiling and why it is a hard human-noise limit.

### Design Decisions
- CLIP embeddings are L2-normalized before saving (unit-sphere → consistent scale for the
  fusion gate's reliability softmax).
- Sequence length T=16 matched exactly between the original cache and CLIP extraction,
  so the merge is a simple array swap with no interpolation.
- Cache signature validation bypassed only for the custom v3 merged cache; the original
  extractor's strong cache contract remains intact for the v2 cache.
- Recommended fine-tuned hyperparams for overnight run: `embedding_dim=128`,
  `attention_dim=64`, `temporal_hidden_dim=128`, `fusion_hidden_dim=256`,
  `lr=3e-4`, `weight_decay=1e-3`, `epochs=40`.

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
