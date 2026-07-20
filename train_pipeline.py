"""
LIRIS-ACCEDE-only training pipeline for video sentiment classification.
"""

import argparse
import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
from scene_motion_llm.train import SceneMotionTrainer
from scene_motion_llm.utils.config import (
    ATTENTION_DIM,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    FUSION_DIM,
    LEARNING_RATE,
    MAX_FRAMES,
    NUM_CLASSES,
    NUM_EPOCHS,
    OUTPUT_PATH,
    SPATIAL_FEATURE_DIM,
    TEMPORAL_HIDDEN_DIM,
    LIRIS_ACCEDE_PATH,
    WEIGHT_DECAY,
)
from scene_motion_llm.utils.dataset import create_data_loaders, VideoFrameDataset

# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# STAGE 2: SENTIMENT FINE-TUNING
# ============================================

class SentimentFinetuningTrainer:
    """Train the raw-video model only on official LIRIS-ACCEDE labels."""

    def __init__(self, device="cuda", checkpoint_dir="./checkpoints", output_dir="./outputs"):
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Sentiment Fine-tuning Trainer initialized (Device: {self.device})")

    def train_accede(self, args, pretrained_model=None):
        """Step 4: Fine-tune on LIRIS-ACCEDE"""
        logger.info("=" * 60)
        logger.info("STEP 4: Fine-tuning on LIRIS-ACCEDE Sentiment")
        logger.info("=" * 60)

        train_loader, val_loader, test_loader, dataset = create_data_loaders(
            dataset_type="liris_accede",
            dataset_path=args.liris_accede_path,
            labels_path=args.labels_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            compute_flow=args.compute_flow
        )

        if pretrained_model is None:
            model = SceneMotionLLMModel(
                spatial_feature_dim=SPATIAL_FEATURE_DIM,
                motion_feature_dim=2,
                temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
                attention_dim=ATTENTION_DIM,
                fusion_dim=FUSION_DIM,
                num_classes=NUM_CLASSES,
                max_frames=MAX_FRAMES,
                dropout=0.3,
                pretrained=True
            )
        else:
            model = pretrained_model

        checkpoint_dir = self.checkpoint_dir / "stage2_accede"
        output_dir = self.output_dir / "stage2_accede"

        trainer = SceneMotionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=self.device,
            checkpoint_dir=str(checkpoint_dir),
            output_dir=str(output_dir)
        )

        trainer.setup_training(
            learning_rate=args.learning_rate,
            weight_decay=WEIGHT_DECAY
        )

        results = trainer.train(
            num_epochs=args.epochs,
            early_stopping=False
        )

        # Report held-out performance using the same checkpoint that inference
        # will load, rather than the final (possibly overfit) epoch in memory.
        best_checkpoint = checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            trainer.load_checkpoint(str(best_checkpoint))
            results["test_metrics"] = trainer.test()

        logger.info(f"LIRIS-ACCEDE Training Results: {results}")
        return model, results


# ============================================
# STAGE 3: LLM REASONING LAYER
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
                logger.info("Using mock LLM (local rules-based)")
                self.tokenizer = None
                self.model = None

        except Exception as e:
            logger.warning(f"Could not load LLM: {e}. Using mock LLM instead.")
            self.use_local_llm = False
            self.tokenizer = None
            self.model = None

    def generate_explanation(self, sentiment_idx, motion_features=None, optical_flow=None):
        """Generate sentiment explanation"""

        sentiment_label = SENTIMENT_CLASSES.get(sentiment_idx, "Neutral")

        # Mock LLM reasoning rules
        explanations = {
            0: [  # Positive
                "The video shows smooth and fluid motion patterns.",
                "Positive energy and uplifting movement detected.",
                "Rhythmic and coordinated motion suggests happiness or contentment.",
            ],
            1: [  # Neutral
                "The video contains neutral motion patterns.",
                "Standard movement without extreme emotional indicators.",
                "Balanced and steady motion characteristics.",
            ],
            2: [  # Negative
                "Rapid and erratic motion patterns detected.",
                "Abrupt motion changes and avoidance behavior observed.",
                "Motion patterns associated with distress or fear.",
            ]
        }

        import random
        explanation = random.choice(explanations.get(sentiment_idx, [
            "Unable to determine sentiment from motion patterns."
        ]))

        # If LLM is available, enhance explanation
        if self.use_local_llm and self.model is not None:
            prompt = f"Generate a brief explanation for {sentiment_label} sentiment in a video: {explanation}"
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(inputs.input_ids, max_length=256)
                enhanced = self.tokenizer.decode(outputs[0])
                return enhanced[:256]  # Truncate to reasonable length
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                return explanation

        return explanation

    def forward(self, sentiment_logits, motion_features=None, optical_flow=None):
        """Forward pass for reasoning"""
        batch_size = sentiment_logits.shape[0]
        sentiments = torch.argmax(sentiment_logits, dim=1)

        explanations = []
        for i in range(batch_size):
            explanation = self.generate_explanation(
                sentiments[i].item(),
                motion_features=motion_features[i] if motion_features is not None else None,
                optical_flow=optical_flow[i] if optical_flow is not None else None
            )
            explanations.append(explanation)

        return sentiments, explanations


# ============================================
# SENTIMENT + LLM PIPELINE
# ============================================

class SentimentLLMPipeline(nn.Module):
    """Complete pipeline: Sentiment Classifier + LLM Reasoning"""

    def __init__(self, sentiment_model, llm_reasoning_layer):
        super().__init__()
        self.sentiment_model = sentiment_model
        self.llm_reasoning = llm_reasoning_layer

    def forward(self, x, optical_flow=None):
        """
        Forward pass combining sentiment prediction and LLM reasoning

        Returns:
            - sentiments: predicted sentiment indices
            - explanations: LLM-generated explanations
            - logits: raw sentiment logits
        """
        logits = self.sentiment_model(x)
        sentiments, explanations = self.llm_reasoning(logits, optical_flow=optical_flow)

        return {
            'sentiments': sentiments,
            'explanations': explanations,
            'logits': logits,
            'confidence': torch.softmax(logits, dim=1)
        }


# ============================================
# MAIN TRAINING PIPELINE
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LIRIS-ACCEDE video sentiment model")

    # Training config
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    # Dataset paths
    parser.add_argument("--liris-accede-path", type=str, default=LIRIS_ACCEDE_PATH)
    parser.add_argument("--labels-path", type=str, default=None,
                        help="Official ACCEDEaffect.txt or ACCEDEranking.txt annotation file")

    # Options
    parser.add_argument("--compute-flow", dest="compute_flow", action="store_true", default=True,
                        help="Compute optical flow for video training (default)")
    parser.add_argument("--no-compute-flow", dest="compute_flow", action="store_false",
                        help="Disable optical-flow features")

    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_PATH)
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Load pretrained checkpoint")

    return parser.parse_args()


def resolve_device(requested_device):
    if requested_device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU")
        return "cpu"
    return requested_device


def run_all_stages(args):
    """Compatibility entrypoint for the single LIRIS training stage."""
    device = resolve_device(args.device)
    logger.info(f"Running LIRIS-ACCEDE-only training on {device}")

    results = {}

    # ============================================
    # STAGE 2: SENTIMENT FINE-TUNING
    # ============================================

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: SENTIMENT FINE-TUNING (LIRIS-ACCEDE)")
    logger.info("=" * 80)

    sentiment_trainer = SentimentFinetuningTrainer(
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )

    # Check if LIRIS-ACCEDE path exists
    if args.liris_accede_path is None or not Path(args.liris_accede_path).exists():
        logger.error(f"LIRIS-ACCEDE path not found: {args.liris_accede_path}")
        logger.error("Cannot proceed with training. LIRIS-ACCEDE dataset is required.")
        raise ValueError(f"LIRIS-ACCEDE dataset not found at {args.liris_accede_path}")
    
    accede_model, accede_results = sentiment_trainer.train_accede(args)
    results['stage2_accede'] = accede_results
    final_model = accede_model
    
    # ============================================
    # SAVE RESULTS
    # ============================================

    output_file = Path(args.output_dir) / "pipeline_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nLIRIS-ACCEDE training completed! Results saved to {output_file}")
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    for stage, result in results.items():
        logger.info(f"{stage}: {result}")

    return final_model, results


def main():
    args = parse_args()
    device = resolve_device(args.device)

    logger.info("LIRIS-ACCEDE-only training pipeline")
    logger.info(f"Device: {device}")
    run_all_stages(args)


if __name__ == "__main__":
    main()
