"""
Inference Script for SceneMotion-LLM with Sentiment + LLM Reasoning

This script loads a trained model and generates sentiment predictions
along with LLM-based explanations for video clips.

Usage:
    python inference_llm.py --video-path <path_to_video> --model-path <path_to_checkpoint> --use-llm
"""

import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import cv2
import numpy as np

from scene_motion_llm.models.feature_sentiment import FeatureSentimentModel
from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
from scene_motion_llm.utils.config import (
    FRAME_SIZE,
    MAX_FRAMES,
    FPS,
    SPATIAL_FEATURE_DIM,
    TEMPORAL_HIDDEN_DIM,
    ATTENTION_DIM,
    FUSION_DIM,
    NUM_CLASSES,
    SENTIMENT_CLASSES
)
from scene_motion_llm.utils.frame_extractor import FrameExtractor, normalize_frame
from scene_motion_llm.utils.optical_flow import OpticalFlowProcessor

# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# LLM REASONING LAYER
# ============================================

class LLMReasoningLayer(nn.Module):
    """LLM Reasoning Layer for generating sentiment explanations"""

    def __init__(self, use_local_llm=False, model_name="microsoft/phi-2"):
        super().__init__()
        self.use_local_llm = use_local_llm
        self.model_name = model_name

        try:
            if use_local_llm:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                logger.info(f"Loading LLM: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                logger.info("Using rule-based reasoning (LLM disabled)")
                self.tokenizer = None
                self.model = None

        except Exception as e:
            logger.warning(f"Could not load LLM: {e}. Using rule-based reasoning instead.")
            self.use_local_llm = False
            self.tokenizer = None
            self.model = None

    def generate_explanation(self, sentiment_idx, motion_magnitude=None, optical_flow_magnitude=None):
        """Generate sentiment explanation based on motion features"""

        sentiment_label = SENTIMENT_CLASSES.get(sentiment_idx, "Neutral")

        # Rule-based explanations with motion context
        explanations_template = {
            0: [  # Positive
                "The video contains smooth, fluid motion patterns with consistent rhythm.",
                "Positive energy detected with balanced and coordinated movements.",
                "Uplifting motion characteristics suggest happiness, contentment, or excitement.",
                "Synchronized movement patterns indicate positive emotional state.",
            ],
            1: [  # Neutral
                "The video shows neutral motion patterns without extreme characteristics.",
                "Balanced movement with neither pronounced energy nor lethargy.",
                "Standard motion patterns without strong emotional indicators.",
                "Moderate motion intensity suggests neutral or calm emotional state.",
            ],
            2: [  # Negative
                "Rapid and erratic motion patterns detected in the video.",
                "Abrupt motion changes and avoidance behaviors observed.",
                "Jerky movements and sudden transitions suggest distress or fear.",
                "Intense or irregular motion patterns indicate negative emotional state.",
            ]
        }

        import random
        base_explanation = random.choice(
            explanations_template.get(sentiment_idx, ["Unable to determine sentiment."])
        )

        # Enhance with motion-based context
        if motion_magnitude is not None:
            if motion_magnitude > 0.7:
                base_explanation += " High motion intensity detected."
            elif motion_magnitude < 0.3:
                base_explanation += " Low motion intensity suggests calm or stationary scenes."

        # If LLM is available, try to enhance
        if self.use_local_llm and self.model is not None:
            try:
                prompt = f"Sentiment: {sentiment_label}. Observation: {base_explanation}. Provide a brief professional analysis:"
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_length=256,
                        temperature=0.7,
                        top_p=0.9
                    )

                enhanced = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return enhanced[:512]  # Truncate to reasonable length

            except Exception as e:
                logger.warning(f"LLM generation failed: {e}. Using rule-based explanation.")
                return base_explanation

        return base_explanation

    def forward(self, sentiment_logits, motion_magnitude=None, optical_flow_magnitude=None):
        """Forward pass for reasoning"""
        batch_size = sentiment_logits.shape[0]
        sentiments = torch.argmax(sentiment_logits, dim=1)
        confidences = torch.softmax(sentiment_logits, dim=1)

        results = []
        for i in range(batch_size):
            motion_mag = motion_magnitude[i] if motion_magnitude is not None else None
            flow_mag = optical_flow_magnitude[i] if optical_flow_magnitude is not None else None

            explanation = self.generate_explanation(
                sentiments[i].item(),
                motion_magnitude=motion_mag,
                optical_flow_magnitude=flow_mag
            )

            results.append({
                'sentiment_idx': sentiments[i].item(),
                'sentiment_label': SENTIMENT_CLASSES.get(sentiments[i].item(), "Neutral"),
                'confidence': confidences[i].cpu().numpy().tolist(),
                'explanation': explanation
            })

        return results


# ============================================
# SENTIMENT + LLM INFERENCE
# ============================================

class SentimentInferenceEngine:
    """Complete inference engine for sentiment prediction with LLM reasoning"""

    def __init__(self, model, llm_reasoning, device="cuda"):
        self.model = model
        self.llm_reasoning = llm_reasoning
        self.device = device
        self.model.eval()
        self.llm_reasoning.eval()

        self.frame_extractor = FrameExtractor(fps=FPS, frame_size=FRAME_SIZE)
        self.flow_processor = OpticalFlowProcessor()

        logger.info("Sentiment Inference Engine initialized")

    def process_video(self, video_path, return_frames=False):
        """Process single video and generate sentiment + explanation"""

        logger.info(f"Processing video: {video_path}")

        # Extract frames
        frames = self.frame_extractor.extract_frames(video_path, output_dir=None)

        if len(frames) == 0:
            logger.error(f"No frames extracted from {video_path}")
            return None

        # Normalize and prepare frames
        frames_list = [
            normalize_frame(cv2.resize(frame, FRAME_SIZE))
            for frame in frames
        ]

        # Pad or trim to max frames
        if len(frames_list) >= MAX_FRAMES:
            frames_tensor = torch.from_numpy(np.array(frames_list[:MAX_FRAMES], dtype=np.float32))
        else:
            padded = frames_list + [np.zeros_like(frames_list[0])] * (MAX_FRAMES - len(frames_list))
            frames_tensor = torch.from_numpy(np.array(padded, dtype=np.float32))

        frames_tensor = frames_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)

        # Compute optical flow
        optical_flow_tensor = None
        motion_magnitude = None

        if len(frames_list) >= 2:
            optical_flow = self.flow_processor.compute_optical_flow(np.array(frames_list))
            optical_flow_tensor = torch.from_numpy(optical_flow).unsqueeze(0).to(self.device)
            motion_magnitude = torch.tensor(
                [np.mean(np.abs(optical_flow))],
                dtype=torch.float32
            ).to(self.device)

        # Forward pass
        with torch.no_grad():
            sentiment_logits = self.model(frames_tensor)

            # Get LLM reasoning
            results = self.llm_reasoning(
                sentiment_logits,
                motion_magnitude=motion_magnitude
            )

        logger.info(f"Sentiment: {results[0]['sentiment_label']}")
        logger.info(f"Confidence: {max(results[0]['confidence']):.2%}")
        logger.info(f"Reasoning: {results[0]['explanation'][:100]}...")

        output = {
            'video_path': str(video_path),
            'num_frames': len(frames_list),
            'results': results[0]
        }

        if return_frames:
            output['frames'] = frames_list

        return output

    def process_batch(self, video_paths):
        """Process multiple videos"""

        results = []

        for video_path in video_paths:
            try:
                result = self.process_video(video_path)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                continue

        return results


# ============================================
# MAIN INFERENCE FUNCTION
# ============================================

def load_model(checkpoint_path, device="cuda"):
    """Load trained sentiment model from checkpoint"""

    logger.info(f"Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer model type and load accordingly
    model = FeatureSentimentModel(
        input_dim=2048,  # Default MOSEI feature dimension
        hidden_dim=TEMPORAL_HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
        num_classes=NUM_CLASSES,
        dropout=0.3
    )

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference with Sentiment + LLM Reasoning"
    )

    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to video file or directory with videos"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use local LLM for enhanced reasoning (requires transformers)"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./inference_results.json",
        help="Path to save inference results"
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use"
    )

    parser.add_argument(
        "--batch-process",
        action="store_true",
        help="Process all videos in directory"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(args.model_path, device=device)

    # Initialize LLM reasoning
    llm_reasoning = LLMReasoningLayer(
        use_local_llm=args.use_llm,
        model_name="microsoft/phi-2"
    )
    llm_reasoning.to(device)

    # Create inference engine
    inference_engine = SentimentInferenceEngine(model, llm_reasoning, device=device)

    # Process video(s)
    video_path = Path(args.video_path)

    if args.batch_process and video_path.is_dir():
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []

        for ext in video_extensions:
            video_files.extend(video_path.glob(f"**/{ext}"))

        logger.info(f"Found {len(video_files)} video files")

        results = inference_engine.process_batch(video_files)

    else:
        result = inference_engine.process_video(str(video_path))
        results = [result] if result is not None else []

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 80)

    if results:
        for i, res in enumerate(results):
            print(f"\nVideo {i + 1}: {Path(res['video_path']).name}")
            print(f"  Sentiment: {res['results']['sentiment_label']}")
            print(f"  Confidence: {max(res['results']['confidence']):.2%}")
            print(f"  Explanation: {res['results']['explanation'][:150]}...")
    else:
        print("No results generated")


if __name__ == "__main__":
    main()
