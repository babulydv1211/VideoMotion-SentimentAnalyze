# """
# Configuration file for SceneMotion-LLM project
# """

# import os

# # Dataset Configuration
# DATASET_PATH = "./dataset/CMU-MOSEI"
# LABELS_PATH = os.path.join(DATASET_PATH, "labels", "CMU_MOSEI_Labels.csd")
# VIDEO_PATH = "./videos"
# FRAMES_PATH = "./frames"
# OPTICAL_FLOW_PATH = "./optical_flow"
# CHECKPOINT_PATH = "./checkpoints"
# OUTPUT_PATH = "./outputs"

# # Video Processing
# FRAME_SIZE = (112, 112)
# FPS = 10  # Frames per second to extract
# MAX_FRAMES = 8  # Maximum frames per video
# MIN_FRAMES = 5   # Minimum frames required

# # Model Configuration
# SPATIAL_FEATURE_DIM = 512  # ResNet18 output dimension
# TEMPORAL_HIDDEN_DIM = 256
# ATTENTION_DIM = 128
# FUSION_DIM = 512
# NUM_CLASSES = 3  # Positive, Neutral, Negative

# # Training Configuration
# # BATCH_SIZE = 8
# # NUM_EPOCHS = 50
# # LEARNING_RATE = 1e-3
# # WEIGHT_DECAY = 1e-5
# # PATIENCE = 10  
# # GRADIENT_CLIP = 1.0
# # Training Configuration
# BATCH_SIZE = 1
# NUM_EPOCHS = 100
# LEARNING_RATE = 1e-4
# WEIGHT_DECAY = 1e-5
# PATIENCE = 5
# GRADIENT_CLIP = 1.0

# # Device Configuration
# DEVICE = "cuda"  # or "cpu"

# # Data Split
# TRAIN_SPLIT = 0.7
# VAL_SPLIT = 0.15
# TEST_SPLIT = 0.15

# # Sentiment Labels
# SENTIMENT_CLASSES = {
#     0: "Positive",
#     1: "Neutral",
#     2: "Negative"
# }

# REVERSE_SENTIMENT_CLASSES = {
#     "Positive": 0,
#     "Neutral": 1,
#     "Negative": 2
# }

# # Optical Flow Configuration
# OPTICAL_FLOW_THRESHOLD = 0.5
# MOTION_MAGNITUDE_THRESHOLD = 5.0

# # Visualization
# VISUALIZE_OPTICAL_FLOW = True
# SAVE_ATTENTION_MAPS = True

# # LLM Configuration (Optional)
# USE_LOCAL_LLM = False
# LLM_MODEL_NAME = "microsoft/phi-2"  # or "google/gemma-2b"
# LLM_MAX_LENGTH = 256




import os

# ============================================
# DATASET PATHS
# ============================================

DATASET_ROOT = "./dataset"

UCF101_PATH = os.path.join(DATASET_ROOT, "UCF101")
UCF_CRIME_PATH = os.path.join(DATASET_ROOT, "UCF-CRIME")
KINETICS_PATH = os.path.join(DATASET_ROOT, "Kinetics")
CMU_MOSEI_PATH = os.path.join(DATASET_ROOT, "CMU_MOSEI")

LABELS_PATH = os.path.join(
    CMU_MOSEI_PATH,
    "Labels",
    "CMU_MOSEI_Labels.csd"
)

# ============================================
# OUTPUT PATHS
# ============================================

VIDEO_PATH = "./videos"
FRAMES_PATH = "./frames"
OPTICAL_FLOW_PATH = "./optical_flow"
CHECKPOINT_PATH = "./checkpoints"
OUTPUT_PATH = "./outputs"

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

NUM_CLASSES = 400

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
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

PATIENCE = 5
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