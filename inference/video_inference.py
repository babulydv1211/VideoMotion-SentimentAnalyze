

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

# from utils.frame_extractor import FrameExtractor, normalize_frame
# from utils.optical_flow import OpticalFlowProcessor
from scene_motion_llm.utils.frame_extractor import (
    FrameExtractor,
    normalize_frame
)

from scene_motion_llm.utils.optical_flow import (
    OpticalFlowProcessor
)

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
            checkpoint_task = checkpoint.get("task")
            if checkpoint_task == "cmu_mosei_feature_sentiment":
                raise ValueError(
                    "This checkpoint was trained on CMU-MOSEI feature files, not raw video frames. "
                    "Use a video-model checkpoint for video inference."
                )
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
    def _generate_report(self, sentiment, confidence, probabilities, evidence, frames_count):
        """Explain the model's output without claiming unmeasured emotions as facts."""
        alternatives = sorted(
            ((self.sentiment_labels[i], float(probability)) for i, probability in enumerate(probabilities)),
            key=lambda item: item[1], reverse=True
        )
        runner_up, runner_up_confidence = alternatives[1]
        certainty = "clear" if confidence >= 0.70 else "moderate" if confidence >= 0.50 else "uncertain"
        motion_text = (
            "low motion" if evidence["mean_motion"] < 1.5 else
            "moderate motion" if evidence["mean_motion"] < 5.0 else "high motion"
        )
        transition_text = "stable" if evidence["frame_change"] < 0.08 else "frequently changing"

        return f"""
================ VIDEO ANALYSIS REPORT ================

Video Understanding:
The LIRIS-ACCEDE-trained model analyzed {frames_count} sampled frames using visual appearance and optical-flow motion features.

Observed visual evidence:
- Average optical-flow magnitude: {evidence['mean_motion']:.2f} ({motion_text}).
- Average frame-to-frame change: {evidence['frame_change']:.3f} ({transition_text} visuals).
- Average brightness: {evidence['brightness']:.3f} on a 0-1 scale.

Final Sentiment:
{sentiment} ({confidence:.2%}; {certainty} confidence)

Why this class:
The classifier assigned the largest probability to {sentiment}. The closest alternative is {runner_up} at {runner_up_confidence:.2%}; this is a learned association from LIRIS-ACCEDE labels, not proof of a person's internal emotion.

Conclusion:
This video is predicted as {sentiment.lower()} because its combined appearance and motion pattern most closely matches LIRIS-ACCEDE examples in that class. Treat low-confidence predictions as ambiguous rather than definitive.
======================================================
"""

    @staticmethod
    def _visual_evidence(frames, flow):
        """Small, reproducible descriptive statistics used in the report."""
        normalized = np.asarray([normalize_frame(frame) for frame in frames], dtype=np.float32)
        frame_change = 0.0
        if len(normalized) > 1:
            frame_change = float(np.mean(np.abs(np.diff(normalized, axis=0))))
        magnitude = np.linalg.norm(flow, axis=-1) if flow.size else np.array([0.0])
        return {
            "mean_motion": float(np.mean(magnitude)),
            "frame_change": frame_change,
            "brightness": float(np.mean(normalized)) if len(normalized) else 0.0,
        }

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
        if len(frames) == 0:
            raise ValueError(f"No decodable frames found in {video_path}")

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
        valid_mask = torch.zeros((1, self.max_frames), dtype=torch.bool, device=self.device)
        valid_mask[:, :len(frames)] = True

        # -------------------------
        # OPTICAL FLOW
        # -------------------------
        optical_flow_data = self.optical_flow_processor.compute_optical_flow(frames_normalized)
        flow = optical_flow_data["flow"].astype(np.float32)

        pad_flow = np.zeros((1, *flow.shape[1:]), dtype=np.float32)
        flow = np.vstack([flow, pad_flow])
        if len(flow) < self.max_frames:
            flow = np.vstack((flow, np.zeros((self.max_frames - len(flow), *flow.shape[1:]), dtype=np.float32)))
        else:
            flow = flow[:self.max_frames]

        optical_flow_tensor = torch.from_numpy(flow).float()
        optical_flow_tensor = optical_flow_tensor.permute(0, 3, 1, 2)
        optical_flow_tensor = optical_flow_tensor.unsqueeze(0).to(self.device)

        # -------------------------
        # INFERENCE
        # -------------------------
        with torch.no_grad():
            outputs = self.model(frames_tensor, optical_flow_tensor, mask=valid_mask)

        probs = outputs["probabilities"].cpu().numpy()[0]
        pred = outputs["predicted_class"].cpu().numpy()[0]
        logits = outputs["logits"].cpu().numpy()[0]
        attention = outputs["attention_weights"].cpu().numpy()[0][:len(frames)]

        sentiment = self.sentiment_labels[pred]
        confidence = float(probs[pred])
        evidence = self._visual_evidence(frames, flow[:len(frames)])
        frame_motion = np.linalg.norm(flow[:len(frames)], axis=-1).mean(axis=(1, 2))
        ranked_frames = np.argsort(attention)[::-1][:min(3, len(frames))]
        attention_evidence = [
            {
                "sampled_frame": int(frame_index + 1),
                "attention_weight": float(attention[frame_index]),
                "mean_flow_magnitude": float(frame_motion[frame_index]),
            }
            for frame_index in ranked_frames
        ]
        ranked_classes = np.argsort(probs)[::-1]
        runner_up = int(ranked_classes[1])
        model_evidence = {
            "logits": [float(value) for value in logits],
            "probability_margin": float(probs[pred] - probs[runner_up]),
            "runner_up": self.sentiment_labels[runner_up],
            "top_attended_frames": attention_evidence,
        }

        logger.info(f"Prediction: {sentiment} ({confidence:.2%})")

        # -------------------------
        # REPORT
        # -------------------------
        report = self._generate_report(
            sentiment,
            confidence,
            probs,
            evidence,
            len(frames)
        )

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probs,
            "num_frames": len(frames),
            "evidence": evidence,
            "model_evidence": model_evidence,
            "report": report
        }

    # =====================================================
    # BATCH ANALYSIS
    # =====================================================
    def batch_analyze(self, video_dir, output_dir=None, pattern="*.mp4"):

        video_dir = Path(video_dir)
        files = list(video_dir.glob(pattern))

        results = []

        for f in files:
            try:
                r = self.analyze_video(str(f), output_dir=output_dir)
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
