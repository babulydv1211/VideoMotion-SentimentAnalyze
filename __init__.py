"""
SceneMotion-LLM: Sentiment Analysis from Scene Motion
Complete end-to-end multimodal video sentiment analysis system
"""

__version__ = "2.0.0"
__author__ = "AI Engineer & Deep Learning Researcher"
__description__ = "Deep learning system for multimodal video sentiment analysis using PyTorch, OpenCV, and Transformers"

# Lazy imports to avoid torch loading issues
def __getattr__(name):
    if name == 'SceneMotionLLMModel':
        from .models.sentiment_classifier import SceneMotionLLMModel
        return SceneMotionLLMModel
    elif name == 'SceneMotionInferencer':
        from .inference import SceneMotionInferencer
        return SceneMotionInferencer
    elif name == 'VideoFrameDataset':
        from .utils.dataset import VideoFrameDataset
        return VideoFrameDataset
    elif name == 'create_data_loaders':
        from .utils.dataset import create_data_loaders
        return create_data_loaders
    elif name == 'FrameExtractor':
        from .utils.frame_extractor import FrameExtractor
        return FrameExtractor
    elif name == 'OpticalFlowProcessor':
        from .utils.optical_flow import OpticalFlowProcessor
        return OpticalFlowProcessor
    elif name == 'MOSEIDataset':
        from .utils.dataset import VideoFrameDataset
        return VideoFrameDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'SceneMotionLLMModel',
    'SceneMotionInferencer',
    'VideoFrameDataset',
    'create_data_loaders',
    'FrameExtractor',
    'OpticalFlowProcessor',
    'MOSEIDataset',
]
