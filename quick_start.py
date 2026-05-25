"""
Quick Start Guide for SceneMotion-LLM
Run this to get started with the project
"""

import sys
import os
from pathlib import Path

print("=" * 70)
print("SCENEMOTION-LLM: Quick Start Guide")
print("=" * 70)
print()

# Check dependencies
print("Checking dependencies...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("✗ PyTorch not installed. Run: pip install torch torchvision")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError:
    print("✗ OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

try:
    import streamlit
    print(f"✓ Streamlit installed")
except ImportError:
    print("✗ Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

print()
print("=" * 70)
print("QUICK START OPTIONS")
print("=" * 70)
print()

print("1. TRAIN MODEL (with dummy data - no videos required)")
print("   Command: python main_train.py --use-dummy-data")
print("   This trains on synthetic data to verify the pipeline")
print()

print("2. TRAIN MODEL (with your dataset)")
print("   Command: python main_train.py --dataset-path ./datasets/CMU-MOSEI \\")
print("                                   --frames-path ./frames")
print()

print("3. RUN STREAMLIT WEB APP")
print("   Command: streamlit run app.py")
print("   Then open browser to http://localhost:8501")
print()

print("4. ANALYZE A VIDEO (using trained model)")
print("   Command: python main_inference.py path/to/video.mp4 \\")
print("                                     --checkpoint ./checkpoints/best_model.pt \\")
print("                                     --output-dir ./outputs/inference")
print()

print("5. TEST MODEL ON TEST SET")
print("   Command: python train.py (will test at end of training)")
print()

print("6. VIEW TRAINING NOTEBOOK")
print("   Open: notebooks/SceneMotion_LLM_Tutorial.ipynb in Jupyter")
print()

print("=" * 70)
print("FOLDER STRUCTURE CREATED")
print("=" * 70)
print()

# Check directory structure
dirs_to_check = [
    'models',
    'utils',
    'dataset',
    'checkpoints',
    'outputs',
    'frames',
    'optical_flow',
    'notebooks'
]

all_exist = True
for dir_name in dirs_to_check:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"✓ {dir_name}/")
    else:
        print(f"✗ {dir_name}/ (will be created automatically)")
        all_exist = False

print()
print("=" * 70)
print("PYTHON FILES")
print("=" * 70)
print()

files_to_check = [
    'train.py',
    'test.py',
    'inference.py',
    'app.py',
    'main_train.py',
    'main_inference.py',
    'models/spatial_extractor.py',
    'models/temporal_motion.py',
    'models/attention.py',
    'models/sentiment_classifier.py',
    'utils/config.py',
    'utils/dataset.py',
    'utils/frame_extractor.py',
    'utils/optical_flow.py'
]

all_exist = True
for file_name in files_to_check:
    file_path = Path(file_name)
    if file_path.exists():
        print(f"✓ {file_name}")
    else:
        print(f"✗ {file_name}")
        all_exist = False

print()
print("=" * 70)
print("NEXT STEPS")
print("=" * 70)
print()
print("1. Install dependencies:")
print("   pip install -r requirements.txt")
print()
print("2. Try training with dummy data:")
print("   python main_train.py --use-dummy-data --epochs 5")
print()
print("3. Launch Streamlit app:")
print("   streamlit run app.py")
print()
print("4. (Optional) Prepare your own dataset:")
print("   - Place videos in videos/ folder")
print("   - Create labels JSON file")
print("   - Update DATASET_PATH in utils/config.py")
print()
print("=" * 70)
print("For detailed documentation, see README.md")
print("=" * 70)
