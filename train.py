import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

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

        self.device = (
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

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

        self.best_val_accuracy = 0
        self.patience_counter = 0

        print(f"\nUsing Device: {self.device}")

        if torch.cuda.is_available():

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            print(
                f"CUDA Version: "
                f"{torch.version.cuda}"
            )

    # ============================================
    # SETUP
    # ============================================

    def setup_training(
        self,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    ):

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2
        )

        self.scaler = GradScaler('cuda')

    # ============================================
    # TRAIN EPOCH
    # ============================================

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

            self.optimizer.zero_grad()

            with autocast(
                device_type='cuda',
                enabled=(self.device == 'cuda')
            ):

                outputs = self.model(
                    frames,
                    optical_flow
                )

                logits = outputs['logits']

                loss = self.criterion(
                    logits,
                    labels
                )

            # ============================================
            # BACKPROP
            # ============================================

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

                "gpu_mem":
                    f"{torch.cuda.memory_allocated()/1024**3:.2f}GB"

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

                outputs = self.model(
                    frames,
                    optical_flow
                )

                logits = outputs['logits']

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
        patience=PATIENCE
    ):

        print("\nTraining Started...\n")

        total_start = time.time()

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

            self.scheduler.step(val_acc)

            # ============================================
            # SAVE BEST MODEL
            # ============================================

            if val_acc > self.best_val_accuracy:

                self.best_val_accuracy = val_acc

                self.save_checkpoint(
                    epoch,
                    is_best=True
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

    # ============================================
    # SAVE CHECKPOINT
    # ============================================

    def save_checkpoint(
        self,
        epoch,
        is_best=False
    ):

        checkpoint = {

            'epoch': epoch,

            'model_state_dict':
                self.model.state_dict(),

            'optimizer_state_dict':
                self.optimizer.state_dict()
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