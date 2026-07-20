# SceneMotion-LLM

Video sentiment classification from visual appearance and motion. The active
training path uses LIRIS-ACCEDE labels and produces positive, neutral, or
negative predictions.

## Project layout

```
scene_motion_llm/
├── models/       # Spatial, temporal, attention, and classifier modules
├── inference/    # Video inference implementation
├── training/     # Feature-model training utilities
├── utils/        # Configuration, data loading, frames, and optical flow
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

For tests and code-quality tools:

```powershell
pip install -r scene_motion_llm\requirements-dev.txt
```

## Train on LIRIS-ACCEDE

Place the video clips in `scene_motion_llm/dataset/Liris_Accede/` and the
official `ACCEDEaffect.txt` annotation file in
`scene_motion_llm/dataset/annotations/`.

```powershell
python -m scene_motion_llm.train_pipeline --epochs 100 --labels-path .\scene_motion_llm\dataset\annotations\ACCEDEaffect.txt
```

If annotations are unavailable, the data loader can use the documented LIRIS
clip ordering to produce deterministic labels, but official annotations are
recommended.

## Run inference

```powershell
python -m scene_motion_llm.main_inference .\my_video.mp4 --checkpoint .\scene_motion_llm\checkpoints\stage2_accede\best_model.pt
```

The checkpoint is auto-detected when `--checkpoint` is omitted. Results are
written under `outputs/` unless another output directory is provided.

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
