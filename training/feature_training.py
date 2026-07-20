"""Training loop for CMU-MOSEI feature sentiment classification."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm


class FeatureSentimentTrainer:
    """Train and evaluate a feature-sequence sentiment model."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        checkpoint_dir,
        output_dir,
        num_classes=3
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_classes = num_classes

        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_f1 = -1.0
        self.patience_counter = 0

    def setup_training(self, learning_rate, weight_decay):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
        self.scaler = GradScaler(enabled=(self.device == "cuda"))

    def train(self, num_epochs, patience):
        start = time.time()

        for epoch in range(num_epochs):
            train_loss, train_metrics = self._run_epoch(self.train_loader, training=True)
            val_loss, val_metrics = self._run_epoch(self.val_loader, training=False)
            self.scheduler.step(val_metrics["f1"])

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, "best_mosei_feature_model.pt")
                print("Best CMU-MOSEI sentiment model saved")
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Feature sentiment training completed in {(time.time() - start) / 60:.2f} minutes")

    def _run_epoch(self, loader, training):
        self.model.train(training)
        total_loss = 0.0
        predictions = []
        labels = []

        description = "Feature Training" if training else "Feature Validation"
        iterator = tqdm(loader, desc=description, leave=False)

        for batch in iterator:
            features = batch["features"].to(self.device, non_blocking=True)
            target = batch["label"].to(self.device, non_blocking=True)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                with autocast(device_type="cuda", enabled=(self.device == "cuda")):
                    outputs = self.model(features)
                    logits = outputs["logits"]
                    loss = self.criterion(logits, target)

                if training:
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            total_loss += loss.item()
            batch_predictions = torch.argmax(logits, dim=1)
            predictions.extend(batch_predictions.detach().cpu().numpy())
            labels.extend(target.detach().cpu().numpy())
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / max(len(loader), 1), self._metrics(labels, predictions)

    @torch.no_grad()
    def test(self):
        test_loss, metrics = self._run_epoch(self.test_loader, training=False)
        metrics["loss"] = test_loss

        results_path = self.output_dir / "mosei_feature_test_results.json"
        with open(results_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

        print("\nCMU-MOSEI FEATURE TEST RESULTS")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Confusion matrix: {metrics['confusion_matrix']}")
        print(f"Results saved to: {results_path}")

        return metrics

    def save_checkpoint(self, epoch, filename):
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "input_dim": self.model.input_dim,
                "num_classes": self.model.num_classes,
                "task": "cmu_mosei_feature_sentiment"
            },
            path
        )

    def _metrics(self, labels, predictions):
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
        matrix = confusion_matrix(
            labels,
            predictions,
            labels=list(range(self.num_classes))
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": matrix.tolist(),
            "labels_present": sorted(int(label) for label in set(labels)),
            "num_classes": self.num_classes
        }
