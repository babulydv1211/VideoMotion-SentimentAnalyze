"""
Testing module for SceneMotion-LLM
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneMotionTester:
    """Test and evaluate trained model"""
    
    def __init__(self, model, device='cuda'):
        """
        Initialize tester
        
        Args:
            model (nn.Module): Trained model
            device (str): Device to use
        """
        self.model = model.to(device)
        self.device = device
    
    def test_on_loader(self, test_loader, output_dir='./outputs'):
        """
        Test model on a data loader
        
        Args:
            test_loader (DataLoader): Test data loader
            output_dir (str): Output directory for results
        
        Returns:
            dict: Test results
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_video_ids = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                frames = batch['frames'].to(self.device)
                labels = batch['label'].to(self.device)
                video_ids = batch['video_id']
                
                outputs = self.model(frames)
                logits = outputs['logits']
                probs = outputs['probabilities']
                
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_video_ids.extend(video_ids)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Print results
        logger.info(f"\n{'='*60}")
        logger.info("TEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Weighted Precision: {precision:.4f}")
        logger.info(f"Weighted Recall: {recall:.4f}")
        logger.info(f"Weighted F1-Score: {f1:.4f}")
        
        logger.info(f"\nPer-class Metrics:")
        classes = ['Positive', 'Neutral', 'Negative']
        for i, cls in enumerate(classes):
            logger.info(f"  {cls}: Precision={precision_per_class[i]:.4f}, "
                       f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
        
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"{cm}")
        
        logger.info(f"\nClassification Report:")
        logger.info(classification_report(all_labels, all_preds, target_names=classes))
        
        # Save results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probs.tolist(),
            'video_ids': list(all_video_ids)
        }
        
        # Save to JSON
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot results
        self._plot_results(cm, output_dir)
        
        return results
    
    def _plot_results(self, cm, output_dir):
        """Plot confusion matrix and other visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        classes = ['Positive', 'Neutral', 'Negative']
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Test Set')
        plt.tight_layout()
        plt.savefig(output_dir / 'test_confusion_matrix.png', dpi=150)
        logger.info(f"Saved confusion matrix to {output_dir / 'test_confusion_matrix.png'}")
        plt.close()


def test_single_sample(model, frames, device='cuda'):
    """
    Test model on a single sample
    
    Args:
        model (nn.Module): Model
        frames (torch.Tensor): Input frames (T, 3, H, W) or (1, T, 3, H, W)
        device (str): Device
    
    Returns:
        dict: Predictions
    """
    model.eval()
    
    # Ensure batch dimension
    if len(frames.shape) == 4:
        frames = frames.unsqueeze(0)
    
    frames = frames.to(device)
    
    with torch.no_grad():
        outputs = model(frames)
    
    logits = outputs['logits']
    probs = outputs['probabilities']
    predicted_class = outputs['predicted_class']
    
    classes = ['Positive', 'Neutral', 'Negative']
    
    return {
        'logits': logits.cpu().numpy(),
        'probabilities': probs.cpu().numpy(),
        'predicted_class': predicted_class.cpu().numpy(),
        'predicted_sentiment': classes[predicted_class.item()],
        'confidence': probs.max().item()
    }
