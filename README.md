# SceneMotion-LLM: Affective Video Classification

**SceneMotion-LLM** is a multimodal video affect (sentiment) classification system. It classifies short video clips into **Positive**, **Neutral**, or **Negative** sentiment by processing 5 synchronized pillars of information: audio, color/lighting, spatial context, motion, and temporal change.

## 📊 The Dataset: LIRIS-ACCEDE

We use the official **LIRIS-ACCEDE** dataset, consisting of 9,800 movie clips ranked by human emotional valence. 
Labels are mapped from the official `valenceRank` into a fixed global thirds distribution:

| Class    | Rank range  |
|----------|-------------|
| Negative | 0 – 3265    |
| Neutral  | 3266 – 6532 |
| Positive | 6533 – 9799 |

## 🏆 State-of-the-Art (SOTA) & Benchmark Context

When evaluating models on the LIRIS-ACCEDE dataset for 3-class discrete classification, it is important to understand the benchmark ceiling. State-of-the-art multimodal neural networks (combining Video Transformers, Audio, and Motion) typically achieve a maximum accuracy in the **60% to 65% range**.

**Why is the maximum accuracy cap around 65%?**
Human emotion is inherently subjective. The LIRIS-ACCEDE dataset was annotated via crowdsourcing, meaning the "ground truth" labels are averages of human opinions. Because there is heavy overlap between the "Neutral" class and the extreme classes, and because humans frequently disagree on the affective impact of a video, the data contains unavoidable noise. 

If a machine learning model were to score significantly above 65%, it would imply the model is more consistent at predicting human emotion than humans are at agreeing with each other! Therefore, a 60-65% accuracy score represents a highly successful model that has reached the practical limits of agreement on this dataset.

## 🧠 Architecture: Semantic Five-Pillar Fusion

The canonical model accepts clips up to 15 seconds long. Every temporal window receives 5 feature sequences, which are then combined via a learned reliability gate. 

1. **Audio**: MFCC / PANNs
2. **Color/Lighting**: Global HSV stats
3. **Spatial**: CLIP (ViT-B/32) Semantic Embeddings
4. **Motion**: Dense Optical Flow
5. **Temporal Change**: Scene cut frequency & transition dynamics

A clip with no decodable sound has an exactly-zero audio fusion weight; a black or low-detail video keeps visual pillars available at low reliability. Neither case is hard-coded into a default sentiment label.

## 🚀 Getting Started

The core package and its specific README are located in the `scene_motion_llm/` directory. 

To set up the environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r scene_motion_llm\requirements.txt
```

To run the Streamlit UI:
```powershell
streamlit run scene_motion_llm\app.py
```

## 📁 Repository Structure

```text
sentiment_analysis/
├── scene_motion_llm/           # Main package containing all source code
│   ├── api/                    # FastAPI backend for remote inference
│   ├── inference/              # Scripts for analyzing new videos
│   ├── models/                 # PyTorch neural network architectures
│   ├── training/               # Feature extraction and model training
│   ├── ui/                     # Streamlit frontend application
│   └── utils/                  # Helper functions for video/audio processing
├── checkpoints/                # Saved weights for the best trained models
├── outputs/                    # Exported predictions, evaluation metrics, and logs
├── PROJECT_CONTEXT.md          # Internal development documentation
└── README.md                   # This file
```

## 🎥 Running Inference on New Videos

Once the model is trained, you can analyze the sentiment of any custom MP4 clip. The inference script automatically extracts all 5 pillars (Audio, CLIP Visuals, Motion, etc.) and routes them through the fusion network.

```powershell
# Analyze a custom video
python -m scene_motion_llm.scripts.analyze_semantic_video .\my_video.mp4 `
  --checkpoint .\scene_motion_llm\checkpoints\semantic_five_pillar\best_semantic_five_pillar.pt
```

**What the output looks like:**
The output report includes:
- **Final Label**: Positive, Neutral, or Negative
- **P/N/N class probabilities**
- **Model Confidence score**
- **Learned fusion weights** (which pillars the model relied on most for this specific video)
- **Semantic cues** (e.g., lighting conditions, motion intensity)

## 🌐 Web Interface (UI)

We provide a local Streamlit dashboard for easy, drag-and-drop video analysis.

```powershell
streamlit run scene_motion_llm\app.py
```
This launches a web page where you can upload a video, click "Analyze", and view a breakdown of the 5 pillars alongside the final sentiment prediction.

## 🤝 Contributing & License

This project is open-source. For details on usage, please see the `LICENSE` file. When contributing, please ensure all new features are accompanied by appropriate unit tests.
