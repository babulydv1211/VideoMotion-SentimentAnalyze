import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast
from torch.cuda.amp import GradScaler

import numpy as np
import logging
import time

from tqdm import tqdm

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from scene_motion_llm.utils.config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    DEVICE
)

# ============================================
# LOGGING
# ============================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# ============================================
# GPU OPTIMIZATION
# ============================================

torch.backends.cudnn.benchmark = True

# ============================================
# TRAINER
# ============================================

class SceneMotionTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader=None,
        device=DEVICE,
        checkpoint_dir='./checkpoints',
        output_dir='./outputs'
    ):

        self.device = device
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU")
            self.device = 'cpu'

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.best_val_accuracy = float('-inf')
        self.patience_counter = 0

        print(f"\nUsing Device: {self.device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

    def setup_training(self, learning_rate=1e-4, weight_decay=1e-5):
        try:
            if hasattr(self.train_loader.dataset, 'indices'):
                dataset = self.train_loader.dataset.dataset
                indices = self.train_loader.dataset.indices
                labels = [dataset.labels[i] for i in indices]
            else:
                labels = self.train_loader.dataset.labels
            import numpy as np
            import torch
            class_counts = np.bincount(labels)
            total_samples = len(labels)
            class_counts[class_counts == 0] = 1 
            weights = total_samples / (len(class_counts) * class_counts)
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            import logging
            logging.info(f"Applied Class Weights: {weights.tolist()}")
        except Exception as e:
            self.criterion = torch.nn.CrossEntropyLoss()

        trainable_params = list(self.model.parameters())
        if trainable_params:
            self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2)
        else:
            self.optimizer = None
            self.scheduler = None

        from torch.cuda.amp import GradScaler
        self.scaler = GradScaler(enabled=(self.device == 'cuda'))

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        all_preds = []
        all_labels = []

        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
            leave=False
        )

        for batch_idx, batch in progress_bar:

            frames = batch['frames'].to(
                self.device,
                non_blocking=True
            )

            labels = batch['label'].to(
                self.device,
                non_blocking=True
            )

            optical_flow = None

            if batch.get('optical_flow') is not None:

                optical_flow = (
                    batch['optical_flow'].to(
                        self.device,
                        non_blocking=True
                    )
                )

            if self.optimizer is not None:
                self.optimizer.zero_grad()

            with autocast(
                device_type='cuda',
                enabled=(self.device == 'cuda')
            ):

                audio_features = batch.get('audio_features')
                outputs = self.model(
                    frames,
                    optical_flow,
                    audio_features=audio_features
                )

                logits = outputs['logits'].to(
                    self.device,
                    non_blocking=True
                )

                loss = self.criterion(
                    logits,
                    labels
                )

            # ============================================
            # BACKPROP
            # ============================================

            if self.optimizer is not None and loss.requires_grad:
                self.scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            # ============================================
            # METRICS
            # ============================================

            total_loss += loss.item()

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.detach()
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            # ============================================
            # PROGRESS BAR
            # ============================================

            progress_bar.set_postfix({

                "loss":
                    f"{loss.item():.4f}",

                "gpu_mem": (
                    f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"
                    if self.device == 'cuda' else "n/a"
                )

            })

        avg_loss = (
            total_loss /
            len(self.train_loader)
        )

        accuracy = accuracy_score(
            all_labels,
            all_preds
        )

        return avg_loss, accuracy

    # ============================================
    # VALIDATION
    # ============================================

    @torch.no_grad()
    def validate_epoch(self):

        self.model.eval()

        total_loss = 0

        all_preds = []
        all_labels = []

        progress_bar = tqdm(
            self.val_loader,
            total=len(self.val_loader),
            desc="Validation",
            leave=False
        )

        for batch in progress_bar:

            frames = batch['frames'].to(
                self.device,
                non_blocking=True
            )

            labels = batch['label'].to(
                self.device,
                non_blocking=True
            )

            optical_flow = None

            if batch.get('optical_flow') is not None:

                optical_flow = (
                    batch['optical_flow'].to(
                        self.device,
                        non_blocking=True
                    )
                )

            with autocast(
                device_type='cuda',
                enabled=(self.device == 'cuda')
            ):

                audio_features = batch.get('audio_features')
                outputs = self.model(
                    frames,
                    optical_flow,
                    audio_features=audio_features
                )

                logits = outputs['logits'].to(
                    self.device,
                    non_blocking=True
                )

                loss = self.criterion(
                    logits,
                    labels
                )

            total_loss += loss.item()

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.detach()
                .cpu()
                .numpy()
            )

            all_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
            )

            progress_bar.set_postfix({

                "val_loss":
                    f"{loss.item():.4f}"

            })

        avg_loss = (
            total_loss /
            len(self.val_loader)
        )

        accuracy = accuracy_score(
            all_labels,
            all_preds
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                all_labels,
                all_preds,
                average='weighted',
                zero_division=0
            )
        )

        return (
            avg_loss,
            accuracy,
            precision,
            recall,
            f1
        )

    # ============================================
    # TRAIN LOOP
    # ============================================

    def train(
        self,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        early_stopping=True
    ):

        print("\nTraining Started...\n")

        total_start = time.time()
        history = []
        best_epoch = 0

        for epoch in range(num_epochs):

            print(
                f"\n========== "
                f"EPOCH {epoch+1}/{num_epochs} "
                f"=========="
            )

            epoch_start = time.time()

            train_loss, train_acc = (
                self.train_epoch()
            )

            (
                val_loss,
                val_acc,
                precision,
                recall,
                f1
            ) = self.validate_epoch()

            epoch_time = (
                time.time() - epoch_start
            )

            print(
                f"\nEpoch "
                f"[{epoch+1}/{num_epochs}] | "
                f"Train Loss: "
                f"{train_loss:.4f} | "
                f"Train Acc: "
                f"{train_acc*100:.2f}% | "
                f"Val Loss: "
                f"{val_loss:.4f} | "
                f"Val Acc: "
                f"{val_acc*100:.2f}% | "
                f"F1: "
                f"{f1:.4f} | "
                f"Time: "
                f"{epoch_time:.2f}s"
            )

            if self.scheduler is not None:
                self.scheduler.step(val_acc)

            # ============================================
            # SAVE BEST MODEL
            # ============================================

            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'patience_counter': self.patience_counter,
                'best_val_accuracy': self.best_val_accuracy,
            }
            history.append(epoch_metrics)

            if val_acc > self.best_val_accuracy:

                self.best_val_accuracy = val_acc
                best_epoch = epoch + 1
                epoch_metrics['best_val_accuracy'] = self.best_val_accuracy

                self.save_checkpoint(
                    epoch,
                    is_best=True,
                    metrics=epoch_metrics,
                )

                print(
                    "\nBest Model Saved"
                )

                self.patience_counter = 0

            else:

                self.patience_counter += 1

                print(
                    f"\nPatience: "
                    f"{self.patience_counter}/"
                    f"{patience}"
                )

            # ============================================
            # EARLY STOPPING
            # ============================================

            if (
                early_stopping
                and
                self.patience_counter
                >= patience
            ):

                print(
                    f"\nEarly stopping "
                    f"at epoch "
                    f"{epoch+1}"
                )

                break

        total_time = (
            time.time() - total_start
        )

        print(
            f"\nTraining Completed "
            f"in "
            f"{total_time/60:.2f} minutes"
        )

        return {
            'history': history,
            'best_epoch': best_epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'val_accuracy': history[-1]['val_accuracy'] if history else None,
            'train_accuracy': history[-1]['train_accuracy'] if history else None,
            'final_train_accuracy': history[-1]['train_accuracy'] if history else None,
            'final_val_accuracy': history[-1]['val_accuracy'] if history else None,
            'final_f1': history[-1]['f1'] if history else None,
        }

    # ============================================
    # SAVE CHECKPOINT
    # ============================================

    def save_checkpoint(
        self,
        epoch,
        is_best=False,
        metrics=None,
    ):

        checkpoint = {

            'epoch': epoch,

            'model_state_dict':
                self.model.state_dict(),

            'optimizer_state_dict':
                self.optimizer.state_dict() if self.optimizer is not None else None,
            'num_classes': 3,
            'task': 'liris_accede_video_sentiment',
            'metrics': metrics or {},
        }

        if is_best:

            path = (
                self.checkpoint_dir /
                'best_model.pt'
            )

        else:

            path = (
                self.checkpoint_dir /
                f'checkpoint_{epoch}.pt'
            )

        torch.save(
            checkpoint,
            path
        )

    # ============================================
    # LOAD CHECKPOINT
    # ============================================

    def load_checkpoint(
        self,
        path
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint['model_state_dict']
        )

        if self.optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )

        print(
            f"Checkpoint Loaded: "
            f"{path}"
        )

    # ============================================
    # TEST
    # ============================================

    @torch.no_grad()
    def test(self):

        if self.test_loader is None:

            print(
                "No Test Loader Found"
            )

            return None

        self.model.eval()

        all_preds = []
        all_labels = []

        progress_bar = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc="Testing",
            leave=False
        )

        for batch in progress_bar:

            frames = batch['frames'].to(
                self.device
            )

            labels = batch['label'].to(
                self.device
            )

            optical_flow = None

            if batch.get('optical_flow') is not None:

                optical_flow = (
                    batch['optical_flow'].to(
                        self.device
                    )
                )

            outputs = self.model(
                frames,
                optical_flow
            )

            logits = outputs['logits']

            preds = torch.argmax(
                logits,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

        accuracy = accuracy_score(
            all_labels,
            all_preds
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                all_labels,
                all_preds,
                average='weighted',
                zero_division=0
            )
        )

        print(
            "\n========== "
            "TEST RESULTS "
            "=========="
        )

        print(
            f"Accuracy : "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision: "
            f"{precision:.4f}"
        )

        print(
            f"Recall   : "
            f"{recall:.4f}"
        )

        print(
            f"F1 Score : "
            f"{f1:.4f}"
        )

        cm = confusion_matrix(
            all_labels,
            all_preds
        )

        print("\nConfusion Matrix:")
        print(cm)

        return {

            'accuracy': accuracy,

            'precision': precision,

            'recall': recall,

            'f1': f1
        }
