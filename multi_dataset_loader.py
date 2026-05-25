import torch

from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    random_split
)

from scene_motion_llm.utils.dataset import (
    VideoFrameDataset,
    _collate_fn
)

from scene_motion_llm.utils.config import *

# ============================================
# GPU SETTINGS
# ============================================

NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# ============================================
# FUNCTION
# ============================================

def create_multi_dataset_loaders():

    datasets_config = [
        {
            "dataset_type": "ucf101",
            "dataset_path": UCF101_PATH
        },
        {
            "dataset_type": "ucf_crime",
            "dataset_path": UCF_CRIME_PATH
        },
        {
            "dataset_type": "kinetics400",
            "dataset_path": KINETICS_PATH
        }
    ]

    all_datasets = []

    # ============================================
    # LOAD DATASETS
    # ============================================

    for config in datasets_config:

        dataset = VideoFrameDataset(
            dataset_type=config["dataset_type"],
            dataset_path=config["dataset_path"],
            split="train",
            compute_flow=False
        )

        print(
            f"{config['dataset_type']} "
            f"Samples: {len(dataset)}"
        )

        if len(dataset) > 0:
            all_datasets.append(dataset)

    if len(all_datasets) == 0:

        raise ValueError(
            "No datasets loaded"
        )

    # ============================================
    # COMBINE DATASETS
    # ============================================

    combined_dataset = ConcatDataset(
        all_datasets
    )

    total_size = len(combined_dataset)

    train_size = int(0.7 * total_size)

    val_size = int(0.15 * total_size)

    test_size = (
        total_size -
        train_size -
        val_size
    )

    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset,
        [train_size, val_size, test_size]
    )

    # ============================================
    # DATALOADERS
    # ============================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=_collate_fn
    )

    # ============================================
    # INFO
    # ============================================

    print(f"\nTotal Samples: {total_size}")

    print(f"Train Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(val_dataset)}")
    print(f"Test Samples: {len(test_dataset)}")

    print(f"\nBatch Size: {BATCH_SIZE}")
    print(f"Workers: {NUM_WORKERS}")

    return (
        train_loader,
        val_loader,
        test_loader
    )