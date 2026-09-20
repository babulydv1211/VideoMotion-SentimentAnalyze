"""
Main Inference Script for SceneMotion-LLM
Analyze videos using a trained model
"""

import argparse
import logging
import os
from pathlib import Path

import torch

# Heavy optional dependencies (whisper, transformers) are imported lazily
# inside the functions that need them so that importing this module just to
# call resolve_checkpoint_path does not require those packages.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def resolve_checkpoint_path(project_root=None, requested_checkpoint=None):
    """Resolve the best available checkpoint path for inference."""
    candidates = []

    if requested_checkpoint:
        candidates.append(Path(requested_checkpoint))

    roots = []
    if project_root is not None:
        roots.append(Path(project_root))

    roots.extend([
        Path.cwd(),
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[1],
    ])

    for root in roots:
        for relative_path in [
            Path("checkpoints") / "stage2_accede" / "best_model.pt",
            Path("checkpoints") / "best_model.pt",
            Path("scene_motion_llm") / "checkpoints" / "stage2_accede" / "best_model.pt",
            Path("scene_motion_llm") / "checkpoints" / "best_model.pt",
            Path("scene_motion_llm") / "checkpoints" / "checkpoint_epoch_4.pt",
            Path("scene_motion_llm") / "checkpoints" / "checkpoint_epoch_0.pt",
        ]:
            candidates.append(root / relative_path)

    package_checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    if package_checkpoint_dir.exists():
        for checkpoint in sorted(package_checkpoint_dir.glob("*.pt")):
            candidates.append(checkpoint)

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return str(resolved)

    if requested_checkpoint:
        return str(Path(requested_checkpoint).expanduser())

    return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze video sentiment using SceneMotion-LLM')

    parser.add_argument('video', type=str,
                        help='Path to video file or directory for batch processing')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a LIRIS-ACCEDE video-model checkpoint (auto-detected when omitted)')
    parser.add_argument('--output-dir', type=str, default='./outputs/inference',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--max-frames', type=int, default=30,
                        help='Maximum frames to process')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for extraction')
    parser.add_argument('--batch', action='store_true',
                        help='Process directory as batch')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                        help='File pattern for batch processing')

    return parser.parse_args()


def main():
    """Main inference function"""
    # Lazy import heavyweight dependencies only when actually running inference
    from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
    from scene_motion_llm.inference import SceneMotionInferencer

    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(
        project_root=Path.cwd(),
        requested_checkpoint=args.checkpoint
    )

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    logger.info(f"Using device: {args.device}")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "No trained LIRIS-ACCEDE checkpoint was found. Train first, or pass "
            "--checkpoint checkpoints/stage2_accede/best_model.pt."
        )
    else:
        logger.info(f"Using checkpoint: {checkpoint_path}")

    logger.info("Creating model...")
    model = SceneMotionLLMModel(
        spatial_feature_dim=512,
        motion_feature_dim=2,
        temporal_hidden_dim=256,
        attention_dim=128,
        fusion_dim=512,
        num_classes=3,
        max_frames=args.max_frames,
        pretrained=True
    )

    logger.info("Initializing inferencer...")
    inferencer = SceneMotionInferencer(
        model=model,
        checkpoint_path=checkpoint_path,
        device=args.device,
        max_frames=args.max_frames,
        fps=args.fps
    )

    if args.batch and os.path.isdir(args.video):
        logger.info(f"Processing directory: {args.video}")
        results = inferencer.batch_analyze(
            video_dir=args.video,
            output_dir=args.output_dir,
            pattern=args.pattern
        )
        logger.info("\n" + "="*70)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("="*70)
        for result in results:
            if 'error' not in result:
                logger.info(f"{Path(result['video']).name}: {result['sentiment']} ({result['confidence']:.1%})")
            else:
                logger.error(f"{Path(result['video']).name}: ERROR - {result['error']}")

    elif os.path.isfile(args.video):
        logger.info(f"Processing video: {args.video}")
        result = inferencer.analyze_video(
            video_path=args.video,
            output_dir=args.output_dir,
            save_frames=True
        )
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*70)
        logger.info(f"Sentiment: {result['sentiment']}")
        logger.info(f"Confidence: {result['confidence']:.1%}")
        logger.info(f"Frames analyzed: {result['num_frames']}")
        logger.info(f"Results saved to: {args.output_dir}")

    else:
        logger.error(f"File or directory not found: {args.video}")


if __name__ == '__main__':
    main()
