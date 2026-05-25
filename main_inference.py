"""
Main Inference Script for SceneMotion-LLM
Analyze videos using a trained model
"""

import torch
import argparse
import logging
import os
from pathlib import Path

from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
from scene_motion_llm.inference import SceneMotionInferencer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze video sentiment using SceneMotion-LLM')
    
    parser.add_argument('video', type=str,
                       help='Path to video file or directory for batch processing')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
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
    args = parse_args()
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.warning(f"Checkpoint not found: {args.checkpoint}")
        logger.info("Using randomly initialized model")
    
    # Create model
    logger.info("Creating model...")
    model = SceneMotionLLMModel(
        spatial_feature_dim=2048,
        temporal_hidden_dim=256,
        attention_dim=128,
        fusion_dim=512,
        num_classes=3,
        max_frames=args.max_frames,
        pretrained=True
    )
    
    # Create inferencer
    logger.info("Initializing inferencer...")
    inferencer = SceneMotionInferencer(
        model=model,
        checkpoint_path=args.checkpoint if os.path.exists(args.checkpoint) else None,
        device=args.device,
        max_frames=args.max_frames,
        fps=args.fps
    )
    
    # Process video(s)
    if args.batch and os.path.isdir(args.video):
        logger.info(f"Processing directory: {args.video}")
        results = inferencer.batch_analyze(
            video_dir=args.video,
            output_dir=args.output_dir,
            pattern=args.pattern
        )
        
        # Print summary
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
        return


if __name__ == '__main__':
    main()
