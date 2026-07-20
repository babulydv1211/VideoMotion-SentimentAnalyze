"""Utilities for CMU-MOSEI pre-extracted feature files."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CMUMOSEIFeatureDataset(Dataset):
    """Dataset for pickled CMU-MOSEI feature sequences."""

    def __init__(self, feature_file, input_dim=None, num_classes=3):
        self.feature_file = Path(feature_file)
        self.num_classes = num_classes

        if not self.feature_file.exists():
            raise FileNotFoundError(f"CMU-MOSEI feature file not found: {self.feature_file}")

        with open(self.feature_file, "rb") as fp:
            self.samples = pickle.load(fp)

        if not self.samples:
            raise ValueError(f"No samples found in {self.feature_file}")

        self.input_dim = input_dim or max(int(sample["feature"].shape[1]) for sample in self.samples)
        self.labels = [int(sample["label"]) for sample in self.samples]

        invalid_labels = sorted({label for label in self.labels if label < 0 or label >= num_classes})
        if invalid_labels:
            raise ValueError(
                f"{self.feature_file} contains labels outside 0..{num_classes - 1}: {invalid_labels}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = np.asarray(sample["feature"], dtype=np.float32)
        features = self._fit_feature_dim(features)

        return {
            "features": torch.from_numpy(features),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "sample_id": idx
        }

    def _fit_feature_dim(self, features):
        current_dim = features.shape[1]

        if current_dim == self.input_dim:
            return features

        if current_dim > self.input_dim:
            return features[:, :self.input_dim]

        padding = np.zeros(
            (features.shape[0], self.input_dim - current_dim),
            dtype=np.float32
        )
        return np.concatenate([features, padding], axis=1)


def create_mosei_feature_loaders(
    dataset_path,
    batch_size,
    num_workers=0,
    num_classes=3
):
    root = Path(dataset_path)
    train_dataset = CMUMOSEIFeatureDataset(root / "train.features", num_classes=num_classes)
    input_dim = train_dataset.input_dim
    val_dataset = CMUMOSEIFeatureDataset(root / "val.features", input_dim=input_dim, num_classes=num_classes)
    test_dataset = CMUMOSEIFeatureDataset(root / "test.features", input_dim=input_dim, num_classes=num_classes)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available()
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, input_dim
