
import os
from pathlib import Path

# ============================================
# DATASET PATHS
# ============================================

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = str(PACKAGE_ROOT / "dataset")
DATASET_PATH = DATASET_ROOT

UCF101_PATH = os.path.join(DATASET_ROOT, "UCF101")
KINETICS_PATH = os.path.join(DATASET_ROOT, "Kinetics")
CMU_MOSEI_PATH = os.path.join(DATASET_ROOT, "CMU_MOSEI")
LIRIS_ACCEDE_PATH = os.path.join(DATASET_ROOT, "Liris_Accede")

LABELS_PATH = os.path.join(
    CMU_MOSEI_PATH,
    "Labels",
    "CMU_MOSEI_Labels.csd"
)

# ============================================
# OUTPUT PATHS
# ============================================

VIDEO_PATH = str(PACKAGE_ROOT / "videos")
FRAMES_PATH = str(PACKAGE_ROOT / "frames")
OPTICAL_FLOW_PATH = str(PACKAGE_ROOT / "optical_flow")
CHECKPOINT_PATH = str(PACKAGE_ROOT / "checkpoints")
OUTPUT_PATH = str(PACKAGE_ROOT / "outputs")

# ============================================
# VIDEO CONFIG
# ============================================

FRAME_SIZE = (112, 112)
FPS = 10
MAX_FRAMES = 8
MIN_FRAMES = 5

# ============================================
# MODEL CONFIG
# ============================================

SPATIAL_FEATURE_DIM = 512
TEMPORAL_HIDDEN_DIM = 256
ATTENTION_DIM = 128
FUSION_DIM = 512

NUM_CLASSES = 3
ACTION_NUM_CLASSES = 101
KINETICS_NUM_CLASSES = 400

# ============================================
# TRAINING CONFIG
# ============================================

# BATCH_SIZE = 2
# NUM_EPOCHS = 50

# LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 1e-5

# PATIENCE = 5
# GRADIENT_CLIP = 1.0
BATCH_SIZE = 16
NUM_EPOCHS = 100

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# LIRIS-ACCEDE training always completes the requested epoch count.  This is
# retained only for compatibility with older callers that explicitly opt in to
# early stopping.
PATIENCE = 5
RANDOM_SEED = 42
GRADIENT_CLIP = 1.0

NUM_WORKERS = 4
PIN_MEMORY = True

# ============================================
# DEVICE
# ============================================

DEVICE = "cuda"

# ============================================
# SPLIT
# ============================================

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================
# SENTIMENT LABELS
# ============================================

SENTIMENT_CLASSES = {
    0: "Positive",
    1: "Neutral",
    2: "Negative"
}

REVERSE_SENTIMENT_CLASSES = {
    "Positive": 0,
    "Neutral": 1,
    "Negative": 2
}

# ============================================
# OPTICAL FLOW
# ============================================

OPTICAL_FLOW_THRESHOLD = 0.5
MOTION_MAGNITUDE_THRESHOLD = 5.0

VISUALIZE_OPTICAL_FLOW = True
SAVE_ATTENTION_MAPS = True

# ============================================
# LLM
# ============================================

USE_LOCAL_LLM = False
LLM_MODEL_NAME = "microsoft/phi-2"
LLM_MAX_LENGTH = 256
