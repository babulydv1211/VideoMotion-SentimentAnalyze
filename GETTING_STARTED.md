# Getting Started with SceneMotion-LLM

Welcome to SceneMotion-LLM! This guide will help you get started with the video sentiment analysis system.

## 📋 Prerequisites

- **Python 3.8+**
- **CUDA 11.8+** (optional but recommended for GPU support)
- **4GB+ RAM** (8GB+ recommended)
- **50GB+ disk space** (for dataset and models)

## 🔧 Installation

### Step 1: Clone or Setup Project

```bash
cd scene_motion_llm
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or run the setup script:

```bash
python setup.py
```

### Step 4: Verify Installation

```bash
python quick_start.py
```

You should see ✓ marks for all dependencies.

## 🚀 Quick Start (30 seconds)

### Option 1: Try with Dummy Data (No Videos Needed)

```bash
python main_train.py --use-dummy-data --epochs 5 --batch-size 4
```

This trains the model on synthetic data to verify the pipeline works.

### Option 2: Launch Web Interface

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 📚 Full Tutorial

### 1. View Examples

```bash
python examples.py
```

This shows the model architecture and configuration.

### 2. Interactive Notebook

Open the Jupyter notebook for step-by-step learning:

```bash
jupyter notebook notebooks/SceneMotion_LLM_Tutorial.ipynb
```

### 3. Training from Scratch

#### Prepare Your Data

Place videos in appropriate directories:

```
dataset/
├── train/        # Training videos
├── test/         # Test videos
└── labels/       # Sentiment labels (JSON or txt files)
```

Create a labels file (labels.json):

```json
{
  "video_id_1": 0,
  "video_id_2": 1,
  "video_id_3": 2
}
```

Where: 0=Positive, 1=Neutral, 2=Negative

#### Extract Frames

```bash
python -c "
from utils.frame_extractor import FrameExtractor
import os

extractor = FrameExtractor(fps=10, frame_size=(224, 224))
video_dir = 'dataset/train'

for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_file)
        output_dir = os.path.join('frames', video_file[:-4])
        extractor.extract_frames(video_path, output_dir)
"
```

#### Train Model

```bash
python main_train.py \
    --dataset-path ./dataset \
    --frames-path ./frames \
    --labels-path ./dataset/labels/labels.json \
    --epochs 50 \
    --batch-size 8
```

#### Monitor Training

Training will produce:
- `checkpoints/best_model.pt` - Best model weights
- `outputs/training_curves.png` - Loss and accuracy plots
- `outputs/confusion_matrix.png` - Training confusion matrix

### 4. Test Trained Model

```bash
python test.py
```

This evaluates the model on the test set and generates:
- `outputs/test_results.json` - Detailed metrics
- `outputs/test_confusion_matrix.png` - Test confusion matrix

### 5. Analyze Custom Videos

#### Single Video

```bash
python main_inference.py path/to/video.mp4 \
    --checkpoint ./checkpoints/best_model.pt \
    --output-dir ./outputs/inference/video_1
```

Output will include:
- Sentiment prediction (Positive/Neutral/Negative)
- Confidence score
- Motion analysis report
- Optical flow visualization
- Results JSON file

#### Batch Processing

```bash
python main_inference.py ./videos \
    --batch \
    --pattern '*.mp4' \
    --output-dir ./outputs/batch
```

Results saved to: `./outputs/batch/batch_results.json`

## 🎯 Common Tasks

### Change Configuration

Edit `utils/config.py` to customize:

```python
# Frame processing
FRAME_SIZE = (224, 224)
FPS = 10
MAX_FRAMES = 30

# Model
TEMPORAL_HIDDEN_DIM = 256
FUSION_DIM = 512

# Training
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
```

### Use CPU Instead of GPU

```bash
python main_train.py --use-dummy-data --device cpu
```

### Fine-tune Pretrained Model

```bash
python main_train.py \
    --dataset-path ./dataset \
    --resume ./checkpoints/best_model.pt \
    --learning-rate 1e-4 \
    --epochs 10
```

### Generate Predictions for Multiple Videos

```python
from inference import SceneMotionInferencer
from models.sentiment_classifier import SceneMotionLLMModel

model = SceneMotionLLMModel()
inferencer = SceneMotionInferencer(
    model,
    checkpoint_path='checkpoints/best_model.pt'
)

# Batch analyze
results = inferencer.batch_analyze(
    video_dir='./videos',
    output_dir='./outputs/batch'
)
```

## 🔍 Understanding Output Files

### Training Output

```
outputs/
├── training_curves.png          # Loss and accuracy curves
├── confusion_matrix.png         # Training confusion matrix
└── [checkpoints/]
    └── best_model.pt           # Model weights
```

### Inference Output

```
outputs/inference/video_name/
├── analysis_report.txt          # Detailed analysis
├── results.json                 # Structured results
├── frames/                      # Extracted frames
├── optical_flow/                # Motion visualizations
└── batch_results.json          # (For batch processing)
```

### Analysis Report Example

```
==============================================================================
SCENE MOTION LLM - VIDEO SENTIMENT ANALYSIS REPORT
==============================================================================

VIDEO INFORMATION
-----------------
File: sample.mp4
Total Frames Analyzed: 30

MOTION ANALYSIS
---------------
• High motion detected - dynamic activity or rapid movements
• Variable motion pattern - changing dynamics
• Dense motion - 75.2% of scene has significant motion

SENTIMENT PREDICTION
--------------------
Predicted Sentiment: NEGATIVE
Confidence Score: 89.3%

Probability Distribution:
  Positive: 5.2%
  Neutral:  5.5%
  Negative: 89.3%
==============================================================================
```

## 🚨 Troubleshooting

### GPU Memory Error

```bash
# Reduce batch size
python main_train.py --batch-size 2

# Reduce max frames
# Edit in utils/config.py: MAX_FRAMES = 15
```

### Frames Not Extracted

```bash
# Check video format is supported (MP4, AVI, MOV, etc.)
# Check video is not corrupted
# Increase FPS extraction interval if too slow
```

### Model Not Training

```bash
# Check data is in correct format
# Verify frame images exist
# Check GPU/CPU compatibility
# Try with dummy data first: python main_train.py --use-dummy-data
```

### Streamlit App Not Loading

```bash
# Clear cache
streamlit cache clear

# Run with verbose mode
streamlit run app.py --logger.level=debug
```

## 📊 Model Performance Tips

### Improve Accuracy

1. **More data**: Collect more diverse videos
2. **Fine-tune longer**: Increase `NUM_EPOCHS`
3. **Lower learning rate**: `LEARNING_RATE = 5e-4`
4. **Data augmentation**: Add frame transformations
5. **Ensemble**: Train multiple models

### Speed Up Training

1. **Reduce frames**: `MAX_FRAMES = 15`
2. **Larger batch size**: `BATCH_SIZE = 16` (if GPU allows)
3. **Lower FPS**: `FPS = 5`
4. **Mixed precision**: Use `torch.cuda.amp`

### Better Results

1. **Longer videos**: Provide videos with sufficient motion
2. **Clear labels**: Ensure accurate sentiment labeling
3. **Diverse scenes**: Include various environments and activities
4. **Clean data**: Remove corrupted or unclear videos

## 📖 Additional Resources

- **Main README**: [README.md](README.md)
- **Jupyter Tutorial**: [notebooks/](notebooks/)
- **Examples**: [examples.py](examples.py)
- **Configuration**: [utils/config.py](utils/config.py)
- **PyTorch Documentation**: https://pytorch.org/
- **OpenCV Documentation**: https://opencv.org/

## 🐛 Report Issues

If you encounter bugs or issues:

1. Check the troubleshooting section above
2. Review error messages carefully
3. Check GPU/CUDA compatibility
4. Try with dummy data first
5. Search in README.md for solutions

## 💡 Tips & Tricks

### Development Mode

```bash
# Run with detailed logging
python main_train.py --use-dummy-data 2>&1 | tee training.log
```

### Interactive Testing

```python
# In Python console
from models.sentiment_classifier import SceneMotionLLMModel
import torch

model = SceneMotionLLMModel()
dummy_frames = torch.randn(1, 30, 3, 224, 224)
output = model(dummy_frames)
print(output['probabilities'])
```

### Visualize Model Architecture

```bash
# In notebook or script
from torchviz import make_dot
import torch

model = SceneMotionLLMModel()
x = torch.randn(1, 30, 3, 224, 224)
y = model(x)
```

## 🎓 Next Steps

After getting comfortable with the basics:

1. **Train on full dataset**: Prepare CMU-MOSEI or custom dataset
2. **Fine-tune model**: Optimize for your specific use case
3. **Deploy model**: Use inference API for production
4. **Add features**: Implement additional motion descriptors
5. **Research**: Experiment with different architectures

## ❓ FAQ

**Q: Can I use my own videos?**
A: Yes! Place videos in `./videos/` or specify path in inference script.

**Q: How long does training take?**
A: ~5 minutes per epoch on GPU, ~30 minutes on CPU (depending on dataset size).

**Q: Can I use pretrained weights?**
A: Yes, the ResNet50 backbone uses ImageNet pretrained weights.

**Q: What's the expected accuracy?**
A: Typically 75-85% on balanced sentiment datasets.

**Q: Can I export the model?**
A: Yes, models are saved as `.pt` files and can be exported to ONNX.

---

**Happy analyzing! 🎉**

For more information, see the [README.md](README.md) file.
