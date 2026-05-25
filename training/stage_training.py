"""Multi-stage dataset training pipeline for SceneMotion-LLM."""

import os
import logging
import torch
import argparse
from pathlib import Path

from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
from scene_motion_llm.utils.config import (
    CHECKPOINT_PATH,
    OUTPUT_PATH,
    MAX_FRAMES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    DEVICE
)
from scene_motion_llm.utils.dataset import create_data_loaders
from scene_motion_llm.train import SceneMotionTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stage_model(num_classes, pretrained=False):
    return SceneMotionLLMModel(
        spatial_feature_dim=512,
        motion_feature_dim=2,
        temporal_hidden_dim=256,
        attention_dim=128,
        fusion_dim=512,
        num_classes=num_classes,
        max_frames=MAX_FRAMES,
        dropout=0.3,
        pretrained=pretrained
    )


def train_stage(dataset_type,
                dataset_path,
                frames_path,
                labels_path,
                output_prefix,
                num_classes,
                epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                stage_name='stage'):
    logger.info(f"Starting {stage_name} with dataset_type={dataset_type}")
    model = create_stage_model(num_classes=num_classes, pretrained=False)

    if DEVICE == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = DEVICE

    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_type=dataset_type,
        dataset_path=dataset_path,
        frames_path=frames_path,
        labels_path=labels_path,
        batch_size=batch_size,
        num_workers=4,
        train_split=0.8,
        val_split=0.1
    )

    checkpoint_dir = Path(CHECKPOINT_PATH) / output_prefix
    output_dir = Path(OUTPUT_PATH) / output_prefix
    trainer = SceneMotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        output_dir=str(output_dir)
    )
    trainer.setup_training(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    trainer.train(num_epochs=epochs, patience=PATIENCE)
    return trainer


def run_stage1_pretraining(args):
    """Stage 1: Pretrain motion encoder on UCF101 and Kinetics400."""
    logger.info("Running Stage 1 Pretraining")
    train_stage(
        dataset_type='ucf101',
        dataset_path=args.ucf101_path,
        frames_path=args.ucf101_frames,
        labels_path=args.ucf101_labels,
        output_prefix='stage1_ucf101',
        num_classes=args.ucf101_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stage_name='stage1_ucf101'
    )
    train_stage(
        dataset_type='kinetics400',
        dataset_path=args.kinetics_path,
        frames_path=args.kinetics_frames,
        labels_path=args.kinetics_labels,
        output_prefix='stage1_kinetics400',
        num_classes=args.kinetics_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stage_name='stage1_kinetics400'
    )


def run_stage2_anomaly_training(args):
    """Stage 2: Train anomaly / motion understanding on UCF-Crime."""
    logger.info("Running Stage 2 Anomaly Training")
    train_stage(
        dataset_type='ucf_crime',
        dataset_path=args.ucfcrime_path,
        frames_path=args.ucfcrime_frames,
        labels_path=args.ucfcrime_labels,
        output_prefix='stage2_ucfcrime',
        num_classes=2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stage_name='stage2_ucfcrime'
    )


def run_stage3_finetuning(args):
    """Stage 3: Fine-tune final sentiment model on CMU-MOSEI."""
    logger.info("Running Stage 3 Fine-tuning")
    train_stage(
        dataset_type='cmu_mosei',
        dataset_path=args.cmu_mosei_path,
        frames_path=args.cmu_mosei_frames,
        labels_path=args.cmu_mosei_labels,
        output_prefix='stage3_mosei',
        num_classes=3,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stage_name='stage3_mosei'
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Run multi-stage training pipeline')

    parser.add_argument('--stage', type=str,
                        choices=['stage1', 'stage2', 'stage3', 'all'],
                        default='all',
                        help='Training stage to run')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs for each stage')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')

    parser.add_argument('--ucf101-path', type=str,
                        default='./dataset/UCF101',
                        help='UCF101 dataset root path')
    parser.add_argument('--ucf101-frames', type=str,
                        default='./frames/ucf101',
                        help='Pre-extracted UCF101 frames path')
    parser.add_argument('--ucf101-labels', type=str,
                        default='./dataset/UCF101/labels.txt',
                        help='UCF101 labels file')
    parser.add_argument('--ucf101-classes', type=int, default=101,
                        help='Number of action classes in UCF101')

    parser.add_argument('--kinetics-path', type=str,
                        default='./dataset/Kinetics',
                        help='Kinetics400 dataset root path')
    parser.add_argument('--kinetics-frames', type=str,
                        default='./frames/kinetics400',
                        help='Pre-extracted Kinetics frames path')
    parser.add_argument('--kinetics-labels', type=str,
                        default='./dataset/Kinetics/labels.txt',
                        help='Kinetics400 labels file')
    parser.add_argument('--kinetics-classes', type=int, default=400,
                        help='Number of action classes in Kinetics400')

    parser.add_argument('--ucfcrime-path', type=str,
                        default='./dataset/UCF-CRIME',
                        help='UCF-Crime dataset root path')
    parser.add_argument('--ucfcrime-frames', type=str,
                        default='./frames/ucfcrime',
                        help='Pre-extracted UCF-Crime frames path')
    parser.add_argument('--ucfcrime-labels', type=str,
                        default='./dataset/UCF-CRIME/labels.txt',
                        help='UCF-Crime labels file')

    parser.add_argument('--cmu-mosei-path', type=str,
                        default='./dataset/CMU-MOSEI',
                        help='CMU-MOSEI dataset root path')
    parser.add_argument('--cmu-mosei-frames', type=str,
                        default='./frames/mosei',
                        help='Pre-extracted CMU-MOSEI frames path')
    parser.add_argument('--cmu-mosei-labels', type=str,
                        default='./dataset/CMU-MOSEI/labels/CMU_MOSEI_Labels.csd',
                        help='CMU-MOSEI labels file path')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage == 'stage1':
        run_stage1_pretraining(args)
    elif args.stage == 'stage2':
        run_stage2_anomaly_training(args)
    elif args.stage == 'stage3':
        run_stage3_finetuning(args)
    else:
        run_stage1_pretraining(args)
        run_stage2_anomaly_training(args)
        run_stage3_finetuning(args)


if __name__ == '__main__':
    main()
