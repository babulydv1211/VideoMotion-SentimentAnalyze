"""
Example Script: Using SceneMotion-LLM for Sentiment Analysis
Demonstrates the complete pipeline from video to sentiment prediction
"""

import torch
import numpy as np
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
from scene_motion_llm.inference import SceneMotionInferencer
from scene_motion_llm.utils.config import *


def example_basic_inference():
    """Example 1: Basic video inference"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Video Inference")
    print("="*70)
    
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = SceneMotionLLMModel(
        spatial_feature_dim=SPATIAL_FEATURE_DIM,
        motion_feature_dim=2,
        temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_CLASSES,
        max_frames=MAX_FRAMES
    )
    
    # Create inferencer
    print("Initializing inferencer...")
    inferencer = SceneMotionInferencer(
        model,
        device=device,
        max_frames=MAX_FRAMES,
        fps=10
    )
    
    # Example video path (you would replace this with an actual video)
    video_path = './videos/sample_video.mp4'
    
    if os.path.exists(video_path):
        print(f"Analyzing video: {video_path}")
        result = inferencer.analyze_video(
            video_path,
            output_dir='./outputs/example_1'
        )
        
        print(f"\nPredicted Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nMotion Analysis:")
        print(result['motion_analysis'])
    else:
        print(f"⚠ Video not found: {video_path}")
        print("Please add a video to ./videos/ folder and try again")


def example_model_architecture():
    """Example 2: Understanding model architecture"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Model Architecture Overview")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = SceneMotionLLMModel(
        spatial_feature_dim=SPATIAL_FEATURE_DIM,
        motion_feature_dim=2,
        temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_CLASSES,
        max_frames=MAX_FRAMES
    )
    
    print("\nModel Components:")
    print("-" * 70)
    print(f"1. Spatial Extractor: ResNet50")
    print(f"   - Output dimension: {SPATIAL_FEATURE_DIM}")
    print(f"2. Motion Projection: Linear layer")
    print(f"   - Input: 2 (optical flow)")
    print(f"   - Output: {TEMPORAL_HIDDEN_DIM}")
    print(f"3. Temporal LSTM: Bidirectional LSTM")
    print(f"   - Hidden dimension: {TEMPORAL_HIDDEN_DIM}")
    print(f"   - Output dimension: {TEMPORAL_HIDDEN_DIM * 2}")
    print(f"4. Temporal Attention: Multi-head attention")
    print(f"   - Attention dimension: {ATTENTION_DIM}")
    print(f"5. Feature Fusion:")
    print(f"   - Input: {SPATIAL_FEATURE_DIM} + {TEMPORAL_HIDDEN_DIM * 2}")
    print(f"   - Output: {FUSION_DIM}")
    print(f"6. Sentiment Classifier: 3-class MLP")
    print(f"   - Classes: Positive, Neutral, Negative")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    
    # Forward pass example
    print(f"\nExample Forward Pass:")
    print(f"- Input shape (batch_size=2, frames=30, channels=3, height=224, width=224):")
    
    dummy_frames = torch.randn(2, 30, 3, 224, 224).to(device)
    model = model.to(device)
    
    with torch.no_grad():
        output = model(dummy_frames)
    
    print(f"  - Output logits shape: {output['logits'].shape}")
    print(f"  - Probabilities shape: {output['probabilities'].shape}")
    print(f"  - Predicted classes: {output['predicted_class'].cpu().numpy()}")
    print(f"  - Example probabilities: {output['probabilities'][0].cpu().numpy()}")


def example_custom_training():
    """Example 3: Training setup"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Training Configuration")
    print("="*70)
    
    print("\nTraining Parameters:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Epochs: {NUM_EPOCHS}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Weight decay: {WEIGHT_DECAY}")
    print(f"- Gradient clip: {GRADIENT_CLIP}")
    print(f"- Early stopping patience: {PATIENCE}")
    print(f"\nData Split:")
    print(f"- Training: {TRAIN_SPLIT*100:.0f}%")
    print(f"- Validation: {VAL_SPLIT*100:.0f}%")
    print(f"- Testing: {TEST_SPLIT*100:.0f}%")
    print(f"\nFrame Processing:")
    print(f"- Frame size: {FRAME_SIZE}")
    print(f"- FPS sampling: {FPS}")
    print(f"- Max frames: {MAX_FRAMES}")
    print(f"- Min frames: {MIN_FRAMES}")
    
    print("\nTo train the model, run:")
    print("  python main_train.py --use-dummy-data --epochs 10")
    print("\nTo train with your dataset:")
    print("  python main_train.py --dataset-path ./datasets/CMU-MOSEI \\")
    print("                        --frames-path ./frames")


def example_batch_inference():
    """Example 4: Batch processing"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Video Processing")
    print("="*70)
    
    print("\nBatch processing analyzes multiple videos at once.")
    print("Place videos in a directory and run:")
    print("\n  python main_inference.py ./videos --batch --pattern '*.mp4'")
    print("\nThis will:")
    print("  1. Find all MP4 files in ./videos/")
    print("  2. Analyze each video")
    print("  3. Save results to ./outputs/inference/batch_results.json")


def main():
    """Run examples"""
    print("\n" + "="*70)
    print("SCENEMOTION-LLM: Example Usage")
    print("="*70)
    
    # Verify setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ Device available: {device.upper()}")
    
    # Run examples
    try:
        example_model_architecture()
        example_custom_training()
        example_batch_inference()
        # Uncomment to test inference:
        # example_basic_inference()
        
        print("\n" + "="*70)
        print("Examples completed!")
        print("="*70)
        print("\nFor more detailed usage, see README.md")
        print("To start training: python main_train.py --use-dummy-data")
        print("To launch app: streamlit run app.py")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
