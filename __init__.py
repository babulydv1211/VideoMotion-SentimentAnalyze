"""
SceneMotion-LLM: Sentiment Analysis from Scene Motion
Complete end-to-end multimodal video sentiment analysis system
"""

__version__ = "1.0.0"
__author__ = "AI Engineer & Deep Learning Researcher"
__description__ = "Deep learning system for multimodal video sentiment analysis using PyTorch, OpenCV, and Transformers"

from .models.sentiment_classifier import SceneMotionLLMModel
from .inference import SceneMotionInferencer
from .utils.dataset import VideoFrameDataset, create_data_loaders
from .utils.frame_extractor import FrameExtractor
from .utils.optical_flow import OpticalFlowProcessor

MOSEIDataset = VideoFrameDataset

__all__ = [
    'SceneMotionLLMModel',
    'SceneMotionInferencer',
    'MOSEIDataset',
    'VideoFrameDataset',
    'create_data_loaders',
    'FrameExtractor',
    'OpticalFlowProcessor'
]
