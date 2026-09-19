import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from scene_motion_llm.train import SceneMotionTrainer


class DummyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        frames = torch.randn(8, 3, 32, 32)
        label = torch.tensor(idx % 3, dtype=torch.long)
        return {"frames": frames, "label": label}


class DummyModel(torch.nn.Module):
    def forward(self, frames, optical_flow=None):
        batch_size = frames.shape[0]
        return {"logits": torch.randn(batch_size, 3)}


class TrainerMetricsTest(unittest.TestCase):
    def test_train_returns_metrics_dict(self):
        train_dataset = DummyDataset()
        val_dataset = DummyDataset()

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        trainer = SceneMotionTrainer(
            model=DummyModel(),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            checkpoint_dir=tempfile.mkdtemp(),
            output_dir=tempfile.mkdtemp(),
        )
        trainer.setup_training(learning_rate=1e-3, weight_decay=0.0)

        metrics = trainer.train(num_epochs=1, patience=1)

        self.assertIsInstance(metrics, dict)
        self.assertIn("val_accuracy", metrics)
        self.assertIn("best_val_accuracy", metrics)


if __name__ == "__main__":
    unittest.main()
