"""
Helper utilities and common functions for SceneMotion-LLM
"""

import os
import torch
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and manage training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
    
    def update(self, **kwargs):
        """Update metrics"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(float(value))
    
    def get_average(self, key):
        """Get average value for a metric"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(self.metrics[key])
        return 0
    
    def get_last(self, key):
        """Get last value for a metric"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return self.metrics[key][-1]
        return 0
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            k: {
                'values': v,
                'average': np.mean(v),
                'latest': v[-1] if v else 0
            }
            for k, v in self.metrics.items()
        }


class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, model, optimizer, epoch, metrics, name='checkpoint'):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        path = self.checkpoint_dir / f'{name}_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        return path
    
    def save_best(self, model, optimizer, epoch, metrics):
        """Save best model"""
        path = self.save(model, optimizer, epoch, metrics, name='best_model')
        return path
    
    def load(self, path, model, optimizer=None, device='cpu'):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def load_best(self, model, optimizer=None, device='cpu'):
        """Load best model"""
        best_path = self.checkpoint_dir / 'best_model_epoch_0.pt'
        if not best_path.exists():
            # Find any best model
            best_models = list(self.checkpoint_dir.glob('best_model_*.pt'))
            if best_models:
                best_path = sorted(best_models)[-1]
            else:
                logger.warning("No best model found")
                return None
        
        return self.load(best_path, model, optimizer, device)
    
    def cleanup(self, keep_best=True, keep_recent=3):
        """Clean up old checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
        
        if keep_recent < len(checkpoints):
            to_remove = checkpoints[:-keep_recent]
            for cp in to_remove:
                os.remove(cp)
                logger.info(f"Removed old checkpoint: {cp}")


class AverageMeter:
    """Compute and store average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'avg: {self.avg:.4f}, val: {self.val:.4f}'


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def save_results(results, output_dir, filename='results.json'):
    """Save results to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, np.integer):
            results_serializable[key] = int(value)
        elif isinstance(value, np.floating):
            results_serializable[key] = float(value)
        else:
            results_serializable[key] = value
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Results saved to: {filepath}")


def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Config loaded from: {config_path}")
    return config


def save_config(config, output_dir, filename='config.json'):
    """Save configuration to JSON file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config saved to: {filepath}")


def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_model_info(model):
    """Print model architecture and parameter info"""
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    print(model)
    
    params = count_parameters(model)
    print("\n" + "="*70)
    print("PARAMETER STATISTICS")
    print("="*70)
    print(f"Total parameters:     {params['total']:>15,}")
    print(f"Trainable parameters: {params['trainable']:>15,}")
    print(f"Frozen parameters:    {params['frozen']:>15,}")
    print("="*70 + "\n")


class ProgressBar:
    """Simple progress bar"""
    
    def __init__(self, total, width=50):
        self.total = total
        self.width = width
        self.current = 0
    
    def update(self, amount=1):
        self.current += amount
    
    def __str__(self):
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        return f'[{bar}] {percent*100:.1f}%'


# Logging configuration
def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
