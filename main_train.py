# """
# Complete Training Pipeline for SceneMotion-LLM
# Run this script to train the model from scratch
# """

# import torch
# import torch.nn as nn
# import logging
# import argparse
# from pathlib import Path
# import numpy as np

# # Import project modules
# from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
# from scene_motion_llm.utils.dataset import create_dummy_loaders, create_data_loaders
# from scene_motion_llm.train import SceneMotionTrainer
# from scene_motion_llm.utils.config import *

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# def parse_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Train SceneMotion-LLM model')
    
#     parser.add_argument('--device', type=str, default='cuda',
#                        choices=['cuda', 'cpu'],
#                        help='Device to use for training')
#     parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
#                        help='Number of training epochs')
#     parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
#                        help='Training batch size')
#     parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
#                        help='Learning rate')
#     parser.add_argument('--patience', type=int, default=PATIENCE,
#                        help='Early stopping patience')
#     parser.add_argument('--dataset-path', type=str, default=DATASET_PATH,
#                        help='Path to dataset')
#     parser.add_argument('--frames-path', type=str, default=FRAMES_PATH,
#                        help='Path to extracted frames')
#     parser.add_argument('--dataset-type', type=str, default='cmu_mosei',
#                        choices=['cmu_mosei', 'ucf101', 'ucf', 'kinetics400', 'ucf_crime'],
#                        help='Type of dataset to load')
#     parser.add_argument('--labels-path', type=str, default=LABELS_PATH,
#                        help='Path to labels')
#     parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_PATH,
#                        help='Directory to save checkpoints')
#     parser.add_argument('--output-dir', type=str, default=OUTPUT_PATH,
#                        help='Directory for outputs')
#     parser.add_argument('--use-dummy-data', action='store_true',
#                        help='Use dummy data for testing (no real videos needed)')
#     parser.add_argument('--resume', type=str, default=None,
#                        help='Path to checkpoint to resume training from')
    
#     return parser.parse_args()


# def main():
#     """Main training function"""
#     args = parse_args()
    
#     # Validate device
#     if args.device == 'cuda' and not torch.cuda.is_available():
#         logger.warning("CUDA not available, falling back to CPU")
#         args.device = 'cpu'
    
#     logger.info(f"Training on device: {args.device}")
#     logger.info(f"PyTorch version: {torch.__version__}")
#     if args.device == 'cuda' and torch.cuda.is_available():
#         torch.cuda.empty_cache()
    
#     # Create data loaders
#     logger.info("Creating data loaders...")
#     if args.use_dummy_data:
#         logger.info("Using dummy data for testing")
#         train_loader, val_loader, test_loader = create_dummy_loaders(
#             batch_size=args.batch_size,
#             num_samples=100
#         )
#         num_classes = NUM_CLASSES
#     else:
#         # Use real dataset if available
#         train_loader, val_loader, test_loader, dataset = create_data_loaders(
#             dataset_type=args.dataset_type,
#             dataset_path=args.dataset_path,
#             frames_path=args.frames_path,
#             labels_path=args.labels_path,
#             batch_size=args.batch_size,
#             num_workers=4,
#             train_split=0.7,
#             val_split=0.15
#         )
#         num_classes = len(getattr(dataset, 'class_to_idx', set()) or set(dataset.labels))
#         if num_classes == 0:
#             num_classes = NUM_CLASSES

#     # Create model
#     logger.info("Creating model...")
#     model = SceneMotionLLMModel(
#         spatial_feature_dim=SPATIAL_FEATURE_DIM,
#         motion_feature_dim=2,
#         temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
#         attention_dim=ATTENTION_DIM,
#         fusion_dim=FUSION_DIM,
#         num_classes=num_classes,
#         max_frames=MAX_FRAMES,
#         dropout=0.3,
#         pretrained=False
#     )
    
#     logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
#     logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
#     # Create trainer
#     logger.info("Initializing trainer...")
#     trainer = SceneMotionTrainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         test_loader=test_loader,
#         device=args.device,
#         checkpoint_dir=args.checkpoint_dir,
#         output_dir=args.output_dir
#     )
    
#     # Setup training
#     trainer.setup_training(
#         learning_rate=args.learning_rate,
#         weight_decay=WEIGHT_DECAY
#     )
    
#     # Resume from checkpoint if specified
#     if args.resume:
#         logger.info(f"Resuming from checkpoint: {args.resume}")
#         trainer.load_checkpoint(args.resume)
    
#     # Train model
#     logger.info("Starting training...")
#     logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    
#     trainer.train(num_epochs=args.epochs, patience=args.patience)
    
#     # Test on test set
#     logger.info("Testing on test set...")
#     test_results = trainer.test()
    
#     logger.info("\n" + "="*70)
#     logger.info("TRAINING COMPLETED")
#     logger.info("="*70)
#     logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
#     logger.info(f"Test F1-Score: {test_results['f1']:.4f}")
#     logger.info(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")  
#     logger.info(f"Results saved to: {args.output_dir}")
    
#     return trainer


# if __name__ == '__main__':
#     main()




# import torch
# import logging

# from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
# from scene_motion_llm.train import SceneMotionTrainer

# from scene_motion_llm.multi_dataset_loader import (
#     train_loader,
#     val_loader,
#     test_loader
# )

# from scene_motion_llm.utils.config import *

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# logger = logging.getLogger(__name__)


# def main():

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     logger.info(f"Training Device: {device}")

#     # ============================================
#     # MODEL
#     # ============================================

#     model = SceneMotionLLMModel(
#         spatial_feature_dim=SPATIAL_FEATURE_DIM,
#         motion_feature_dim=2,
#         temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
#         attention_dim=ATTENTION_DIM,
#         fusion_dim=FUSION_DIM,
#         num_classes=NUM_CLASSES,
#         max_frames=MAX_FRAMES,
#         dropout=0.3,
#         pretrained=False
#     )

#     logger.info(
#         f"Model Parameters: {sum(p.numel() for p in model.parameters())}"
#     )

#     # ============================================
#     # TRAINER
#     # ============================================

#     trainer = SceneMotionTrainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         test_loader=test_loader,
#         device=device,
#         checkpoint_dir=CHECKPOINT_PATH,
#         output_dir=OUTPUT_PATH
#     )

#     trainer.setup_training(
#         learning_rate=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY
#     )

#     # ============================================
#     # TRAIN
#     # ============================================

#     logger.info("Starting Training")

#     trainer.train(
#         num_epochs=NUM_EPOCHS,
#         patience=PATIENCE
#     )

#     # ============================================
#     # TEST
#     # ============================================

#     results = trainer.test()

#     logger.info("=" * 60)
#     logger.info("TRAINING COMPLETED")
#     logger.info("=" * 60)

#     logger.info(f"Accuracy: {results['accuracy']}")
#     logger.info(f"F1 Score: {results['f1']}")

#     logger.info(
#         f"Best model saved at: "
#         f"{CHECKPOINT_PATH}/best_model.pt"
#     )


# if __name__ == "__main__":
#     main()



import torch
import logging

from scene_motion_llm.models.sentiment_classifier import (
    SceneMotionLLMModel
)

from scene_motion_llm.train import (
    SceneMotionTrainer
)

from scene_motion_llm.multi_dataset_loader import (
    create_multi_dataset_loaders
)

from scene_motion_llm.utils.config import *

# ============================================
# LOGGING
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================
# MAIN
# ============================================

def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Training Device: {device}"
    )

    # ============================================
    # LOAD DATASETS
    # ============================================

    train_loader, val_loader, test_loader = (
        create_multi_dataset_loaders()
    )

    # ============================================
    # MODEL
    # ============================================

    model = SceneMotionLLMModel(
        spatial_feature_dim=SPATIAL_FEATURE_DIM,
        motion_feature_dim=2,
        temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_CLASSES,
        max_frames=MAX_FRAMES,
        dropout=0.3,
        weights=None
    )
    
    print("\nModel Created Successfully")
    print("Starting Training Loop...\n")
    
    logger.info(
        f"Model Parameters: "
        f"{sum(p.numel() for p in model.parameters())}"
    )

    # ============================================
    # TRAINER
    # ============================================

    trainer = SceneMotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_dir=CHECKPOINT_PATH,
        output_dir=OUTPUT_PATH
    )
    
    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Validation Loader Batches: {len(val_loader)}")
    print(f"Test Loader Batches: {len(test_loader)}")

    trainer.setup_training(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ============================================
    # TRAINING
    # ============================================

    logger.info(
        "Starting Training..."
    )

    trainer.train(
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )

    # ============================================
    # TESTING
    # ============================================

    logger.info(
        "Running Test Evaluation..."
    )

    results = trainer.test()

    logger.info("=" * 60)

    logger.info(
        "TRAINING COMPLETED"
    )

    logger.info("=" * 60)

    logger.info(
        f"Accuracy: {results['accuracy']}"
    )

    logger.info(
        f"F1 Score: {results['f1']}"
    )

    logger.info(
        f"Best Model Saved: "
        f"{CHECKPOINT_PATH}/best_model.pt"
    )


# ============================================
# START
# ============================================

if __name__ == "__main__":

    main()