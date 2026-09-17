# SceneMotion-LLM

Reliability-aware short-video affect classification using five synchronized
pillars: audio, color/lighting, spatial context, motion, and temporal change.
The canonical model predicts **Positive**, **Neutral**, or **Negative** from
LIRIS-ACCEDE valence ranks. The older visual/motion model remains only as a
legacy baseline.

## Project layout

```
scene_motion_llm/
├── models/       # Legacy baseline and semantic five-pillar fusion model
├── inference/    # Transparent upload inference
├── training/     # Official-split LIRIS semantic training utilities
├── utils/        # Five semantic feature extractors and video helpers
├── api/          # Optional FastAPI application
├── ui/           # Streamlit launcher
├── train_pipeline.py
├── main_inference.py
├── requirements.txt
└── requirements-dev.txt
```

Local datasets, checkpoints, extracted frames, optical flow, outputs, and
virtual environments are intentionally ignored by Git.

## Setup

Run commands from the parent directory of this repository so Python can import
the `scene_motion_llm` package.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r scene_motion_llm\requirements.txt
```

For NVIDIA GPU training on Windows, install the matching CUDA build after the
normal dependencies, then verify that PyTorch can see the GPU:

```powershell
pip install --upgrade --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For tests and code-quality tools:

```powershell
pip install -r scene_motion_llm\requirements-dev.txt
```

## Canonical five-pillar workflow

The model accepts clips of up to 15 seconds. Every window gets five feature
sequences, then a learned reliability gate combines them. A clip with no
decodable sound has `audio = NIL` and an exactly-zero audio fusion weight; a
black or low-detail video keeps visual pillars available at low reliability.
Neither case is hard-coded as a sentiment label.

The supervised target is LIRIS-ACCEDE only. Its `valenceRank` is divided over
the complete 9,800-clip ranking range before any split is read:

| Class | Rank range |
|---|---:|
| Negative | 0–3265 |
| Neutral | 3266–6532 |
| Positive | 6533–9799 |

`ACCEDEsets.txt` supplies the official partitions (`1=train`,
`2=validation`, `0=test`). This avoids the legacy model's random split and
strong Neutral-class shortcut.

First make a reproducible feature cache. The paths are explicit on purpose;
do not copy a local Downloads path into code.

```powershell
$videoDir = ".\scene_motion_llm\dataset\Liris_Accede"
$ranking = ".\scene_motion_llm\dataset\annotations\ACCEDEranking.txt"
$sets = "C:\path\to\LIRIS-ACCEDE-annotations\annotations\ACCEDEsets.txt"
$cache = ".\scene_motion_llm\cache\semantic_five_pillar_v2"

# First check a few clips and their audio-status diagnostics.
python -m scene_motion_llm.training.semantic_five_pillar_training `
  --mode precompute --video-dir $videoDir --ranking-path $ranking `
  --sets-path $sets --cache-dir $cache --limit-per-split 5 `
  --use-pretrained-spatial --allow-model-download

# Then create the complete versioned cache before the full experiment.
python -m scene_motion_llm.training.semantic_five_pillar_training `
  --mode precompute --video-dir $videoDir --ranking-path $ranking `
  --sets-path $sets --cache-dir $cache --use-pretrained-spatial
```

Train after the cache finishes:

```powershell
python -m scene_motion_llm.training.semantic_five_pillar_training `
  --mode train --video-dir $videoDir --ranking-path $ranking `
  --sets-path $sets --cache-dir $cache `
  --checkpoint-dir .\scene_motion_llm\checkpoints\semantic_five_pillar `
  --epochs 25 --batch-size 16 --device cuda --use-pretrained-spatial
```

The checkpoint is selected by validation macro-F1, then the selected checkpoint
is evaluated once on the official test set. It saves model/extractor settings,
train-only normalizer statistics, label rule, split protocol, metrics, and the
final test report together in `best_semantic_five_pillar.pt`.

## Legacy baseline

The old raw visual/motion route is retained for comparisons with previous
experiments. It should not be used to claim the five-pillar model's result.

```powershell
python -m scene_motion_llm.train_pipeline --epochs 100 --labels-path .\scene_motion_llm\dataset\annotations\ACCEDEaffect.txt
```

If annotations are unavailable, the data loader can use the documented LIRIS
clip ordering to produce deterministic labels, but official annotations are
recommended.

## Run semantic inference

```powershell
python -m scene_motion_llm.scripts.analyze_semantic_video .\my_video.mp4 `
  --checkpoint .\scene_motion_llm\checkpoints\semantic_five_pillar\best_semantic_five_pillar.pt
```

The report includes P/N/N probabilities, confidence, audio status, learned
five-pillar weights, reliability, descriptive cues, and temporal windows the
model attended to. Cues describe measurements—not emotion, violence, safety,
or intent labels by themselves.

## Optional interfaces

```powershell
# Streamlit
streamlit run scene_motion_llm\app.py

# FastAPI
uvicorn scene_motion_llm.api.fastapi_app:app --reload
```

## Verify

```powershell
python -m pytest tests -q
```

The tests cover the LIRIS training entry point, trainer metrics, and checkpoint
path resolution.

## Research evaluation

Use official LIRIS annotations to evaluate a checkpoint on a deterministic,
stratified hold-out split. The command saves `evaluation.json` (accuracy,
macro-F1, confusion matrix, and class report) and `predictions.csv`.

```powershell
python -m scene_motion_llm.evaluate_liris `
  --dataset-path .\scene_motion_llm\dataset\Liris_Accede `
  --labels-path .\scene_motion_llm\dataset\annotations\ACCEDEaffect.txt `
  --checkpoint .\scene_motion_llm\checkpoints\stage2_accede\best_model.pt
```
