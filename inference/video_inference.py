


# """
# Inference module for SceneMotion-LLM
# Analyze videos and predict sentiment
# """

# import torch
# import numpy as np
# import os
# import logging
# import json
# from pathlib import Path

# from utils.frame_extractor import FrameExtractor, normalize_frame
# from utils.optical_flow import OpticalFlowProcessor

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class SceneMotionInferencer:

#     def __init__(
#         self,
#         model,
#         checkpoint_path=None,
#         device="cuda",
#         max_frames=30,
#         frame_size=(224, 224),
#         fps=10
#     ):

#         self.device = device
#         self.model = model.to(device)
#         self.model.eval()

#         if checkpoint_path and os.path.exists(checkpoint_path):
#             checkpoint = torch.load(checkpoint_path, map_location=device)
#             self.model.load_state_dict(checkpoint["model_state_dict"])
#             logger.info(f"Loaded checkpoint: {checkpoint_path}")

#         self.max_frames = max_frames
#         self.frame_size = frame_size
#         self.fps = fps

#         self.frame_extractor = FrameExtractor(
#             fps=fps,
#             frame_size=frame_size
#         )

#         self.optical_flow_processor = OpticalFlowProcessor()

#         self.sentiment_labels = {
#             0: "Positive",
#             1: "Neutral",
#             2: "Negative"
#         }

#     # =====================================================
#     # MAIN
#     # =====================================================

#     def analyze_video(self, video_path, output_dir=None, save_frames=False):

#         logger.info("=" * 60)
#         logger.info(f"VIDEO: {video_path}")
#         logger.info("=" * 60)

#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)

#         # -------------------------
#         # FRAME EXTRACTION
#         # -------------------------
#         frames = self.frame_extractor.extract_frames(video_path)

#         frames = frames[:self.max_frames]

#         # -------------------------
#         # NORMALIZE FRAMES
#         # -------------------------
#         frames_normalized = np.array([
#             normalize_frame(f).astype(np.float32)
#             for f in frames
#         ], dtype=np.float32)

#         # (T, H, W, C) -> (T, C, H, W)
#         frames_tensor = torch.from_numpy(frames_normalized).float()
#         frames_tensor = frames_tensor.permute(0, 3, 1, 2)

#         # PAD
#         if len(frames_tensor) < self.max_frames:
#             pad = torch.zeros(
#                 self.max_frames - len(frames_tensor),
#                 3,
#                 self.frame_size[0],
#                 self.frame_size[1],
#                 dtype=torch.float32
#             )
#             frames_tensor = torch.cat([frames_tensor, pad], dim=0)

#         frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

#         # -------------------------
#         # OPTICAL FLOW
#         # -------------------------
#         optical_flow_data = self.optical_flow_processor.compute_optical_flow(frames)

#         flow = optical_flow_data["flow"].astype(np.float32)

#         pad_flow = np.zeros((1, *flow.shape[1:]), dtype=np.float32)
#         flow = np.vstack([flow, pad_flow])

#         optical_flow_tensor = torch.from_numpy(flow).float()
#         optical_flow_tensor = optical_flow_tensor.permute(0, 3, 1, 2)
#         optical_flow_tensor = optical_flow_tensor.unsqueeze(0).to(self.device)

#         # -------------------------
#         # INFERENCE
#         # -------------------------
#         with torch.no_grad():
#             outputs = self.model(frames_tensor, optical_flow_tensor)

#         probs = outputs["probabilities"].cpu().numpy()[0]
#         pred = outputs["predicted_class"].cpu().numpy()[0]

#         sentiment = self.sentiment_labels[pred]
#         confidence = float(probs[pred])

#         logger.info(f"Prediction: {sentiment} ({confidence:.2%})")

#         return {
#             "sentiment": sentiment,
#             "confidence": confidence,
#             "probabilities": probs
#         }

#     # =====================================================
#     # BATCH
#     # =====================================================

#     def batch_analyze(self, video_dir, pattern="*.mp4"):

#         video_dir = Path(video_dir)
#         files = list(video_dir.glob(pattern))

#         results = []

#         for f in files:
#             try:
#                 r = self.analyze_video(str(f))
#                 results.append({
#                     "video": str(f),
#                     **r
#                 })
#             except Exception as e:
#                 results.append({
#                     "video": str(f),
#                     "error": str(e)
#                 })

#         return results


"""
Inference module for SceneMotion-LLM
Analyze videos and predict sentiment + generate human report
"""

import torch
import numpy as np
import os
import logging
import json
from pathlib import Path

from utils.frame_extractor import FrameExtractor, normalize_frame
from utils.optical_flow import OpticalFlowProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneMotionInferencer:

    def __init__(
        self,
        model,
        checkpoint_path=None,
        device="cuda",
        max_frames=30,
        frame_size=(224, 224),
        fps=10
    ):

        self.device = device
        self.model = model.to(device)
        self.model.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint: {checkpoint_path}")

        self.max_frames = max_frames
        self.frame_size = frame_size
        self.fps = fps

        self.frame_extractor = FrameExtractor(
            fps=fps,
            frame_size=frame_size
        )

        self.optical_flow_processor = OpticalFlowProcessor()

        self.sentiment_labels = {
            0: "Positive",
            1: "Neutral",
            2: "Negative"
        }

    # =====================================================
    # REPORT GENERATION
    # =====================================================
    def _generate_report(self, sentiment, confidence, frames_count):

        if sentiment == "Positive":
            mood = "The video shows positive and pleasant activity."
        elif sentiment == "Negative":
            mood = "The video shows negative or intense activity."
        else:
            mood = "The video shows neutral or normal activity."

        return f"""
================ VIDEO ANALYSIS REPORT ================

Video Understanding:
The system analyzed {frames_count} frames using deep learning spatial + motion analysis.

Emotion Analysis:
{mood}

Final Sentiment:
{sentiment}

Confidence Score:
{confidence:.2%}

Conclusion:
The video is classified as {sentiment.lower()} based on visual and motion patterns.
======================================================
"""

    # =====================================================
    # MAIN INFERENCE
    # =====================================================
    def analyze_video(self, video_path, output_dir=None, save_frames=False):

        logger.info("=" * 60)
        logger.info(f"VIDEO: {video_path}")
        logger.info("=" * 60)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # -------------------------
        # FRAME EXTRACTION
        # -------------------------
        frames = self.frame_extractor.extract_frames(video_path)
        frames = frames[:self.max_frames]

        # -------------------------
        # NORMALIZE
        # -------------------------
        frames_normalized = np.array([
            normalize_frame(f).astype(np.float32)
            for f in frames
        ], dtype=np.float32)

        frames_tensor = torch.from_numpy(frames_normalized).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)

        # PAD
        if len(frames_tensor) < self.max_frames:
            pad = torch.zeros(
                self.max_frames - len(frames_tensor),
                3,
                self.frame_size[0],
                self.frame_size[1],
                dtype=torch.float32
            )
            frames_tensor = torch.cat([frames_tensor, pad], dim=0)

        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)

        # -------------------------
        # OPTICAL FLOW
        # -------------------------
        optical_flow_data = self.optical_flow_processor.compute_optical_flow(frames)

        flow = optical_flow_data["flow"].astype(np.float32)

        pad_flow = np.zeros((1, *flow.shape[1:]), dtype=np.float32)
        flow = np.vstack([flow, pad_flow])

        optical_flow_tensor = torch.from_numpy(flow).float()
        optical_flow_tensor = optical_flow_tensor.permute(0, 3, 1, 2)
        optical_flow_tensor = optical_flow_tensor.unsqueeze(0).to(self.device)

        # -------------------------
        # INFERENCE
        # -------------------------
        with torch.no_grad():
            outputs = self.model(frames_tensor, optical_flow_tensor)

        probs = outputs["probabilities"].cpu().numpy()[0]
        pred = outputs["predicted_class"].cpu().numpy()[0]

        sentiment = self.sentiment_labels[pred]
        confidence = float(probs[pred])

        logger.info(f"Prediction: {sentiment} ({confidence:.2%})")

        # -------------------------
        # REPORT
        # -------------------------
        report = self._generate_report(
            sentiment,
            confidence,
            len(frames)
        )

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probs,
            "report": report
        }

    # =====================================================
    # BATCH ANALYSIS
    # =====================================================
    def batch_analyze(self, video_dir, pattern="*.mp4"):

        video_dir = Path(video_dir)
        files = list(video_dir.glob(pattern))

        results = []

        for f in files:
            try:
                r = self.analyze_video(str(f))
                results.append({
                    "video": str(f),
                    **r
                })
            except Exception as e:
                results.append({
                    "video": str(f),
                    "error": str(e)
                })

        return results