# 🎬 SceneMotion-LLM: Sentiment Analysis from Scene Motion

A complete end-to-end multimodal video sentiment analysis system built with PyTorch, OpenCV, and optional Transformers support.

## 🚀 What This Project Does

SceneMotion-LLM analyzes videos to predict one of three sentiment classes:
- **Positive**
- **Neutral**
- **Negative**

It combines:
- video frame extraction
- optical flow motion analysis
- a ResNet-based spatial feature extractor
- temporal sequence modeling with LSTM
- attention mechanisms
- feature fusion
- sentiment classification

## 📁 Repository Structure

```
scene_motion_llm/
├── api/                       # FastAPI backend module
├── dataset/                   # Local datasets for CMU-MOSEI, UCF101, UCF-Crime, Kinetics400
├── inference/                 # Inference engine implementation
├── models/                    # Model architecture components
├── scripts/                   # Helper scripts
├── training/                  # Multi-stage training orchestration
├── ui/                        # UI launcher wrappers
├── utils/                     # Config, data loading, frame extraction, optical flow
├── videos/                    # Sample or uploaded videos
├── frames/                    # Extracted video frames
├── optical_flow/              # Optical flow visualizations
├── checkpoints/               # Saved model checkpoints
├── outputs/                   # Analysis outputs
├── app.py                     # Streamlit application
├── main_train.py              # Training entrypoint
├── main_inference.py          # CLI inference entrypoint
├── train.py                   # Training pipeline class
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## ✅ Setup

### 1. Create and activate a virtual environment

```bash
cd scene_motion_llm
python -m venv venv
# Windows
env\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare datasets

Place dataset files in the local dataset folder structure:
- `scene_motion_llm/dataset/CMU_MOSEI/`
- `scene_motion_llm/dataset/UCF101/`
- `scene_motion_llm/dataset/UCF-CRIME/`
- `scene_motion_llm/dataset/Kinetics/`

If you only use CMU-MOSEI, only the `dataset/CMU_MOSEI/` folder is required.

For UCF101, use `--dataset-type ucf101` or `--dataset-type ucf` and point `--dataset-path` at your UCF101 root folder.

## ▶️ Running the Project

### Training

```bash
python main_train.py
```

For a quick validation run with synthetic data:

```bash
python main_train.py --use-dummy-data
```

### Inference

Analyze a single video:

```bash
python main_inference.py path/to/video.mp4 --checkpoint checkpoints/best_model.pt --output-dir outputs/inference
```

Batch inference on a directory:

```bash
python main_inference.py scene_motion_llm/videos --batch --pattern "*.mp4"
```

### Streamlit UI

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

### FastAPI Backend

```bash
uvicorn api.fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

Edit `scene_motion_llm/utils/config.py` to customize behavior.

Important variables:

```python
FRAME_SIZE = (112, 112)
FPS = 10
MAX_FRAMES = 8
BATCH_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 5
DEVICE = 'cuda'
```

> If CUDA is unavailable, change `DEVICE` to `cpu` or allow the code to fall back automatically.

## 🧠 Model Overview

### Key components
- `models/spatial_extractor.py`: ResNet-based spatial feature extraction
- `models/temporal_motion.py`: LSTM temporal motion modeling
- `models/attention.py`: Attention mechanisms
- `models/sentiment_classifier.py`: Fusion and classification head

### Utility modules
- `utils/frame_extractor.py`: Frame sampling and extraction
- `utils/optical_flow.py`: Optical flow computation
- `utils/dataset.py`: Multi-dataset PyTorch loader
- `inference/video_inference.py`: Video analysis engine

## 📚 Typical Workflow

1. Place raw videos or dataset files in the dataset folder.
2. Train the model using `python main_train.py`.
3. Run inference with `python main_inference.py`.
4. Optionally use `streamlit run app.py` for interactive video analysis.

## 📄 Useful Scripts

- `main_train.py` — training entrypoint
- `main_inference.py` — inference entrypoint
- `app.py` — Streamlit application
- `api/fastapi_app.py` — REST API backend
- `training/stage_training.py` — multi-stage dataset training orchestration

## 📌 Notes

- `train.py` contains the training loop, metrics, checkpoint saving, and plotting.
- `main_train.py` and `main_inference.py` are the recommended CLI entrypoints.
- The project is designed for real video input and optical flow motion analysis.

## 🛠️ Troubleshooting

- Install missing libraries with `pip install -r requirements.txt`.
- Ensure paths in `scene_motion_llm/utils/config.py` point to existing directories.
- If `fastapi` is not installed, add it with `pip install fastapi uvicorn`.

Enjoy using SceneMotion-LLM for video sentiment analysis! 🎬
