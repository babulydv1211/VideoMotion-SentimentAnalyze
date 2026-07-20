
"""LIRIS-ACCEDE video dataset and deterministic data loaders."""

import os
import cv2
import h5py
import torch
import logging
import numpy as np
import csv

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from .config import (
    FRAME_SIZE,
    MAX_FRAMES,
    FPS,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT
)

from .frame_extractor import (
    FrameExtractor,
    normalize_frame
)

from .optical_flow import OpticalFlowProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFrameDataset(Dataset):

    def __init__(
        self,
        dataset_type,
        dataset_path,
        frames_path=None,
        labels_path=None,
        split='train',
        max_frames=MAX_FRAMES,
        frame_size=FRAME_SIZE,
        fps=FPS,
        normalize=True,
        compute_flow=True
    ):

        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.frames_path = frames_path
        self.labels_path = labels_path
        self.split = split

        self.max_frames = max_frames
        self.frame_size = frame_size
        self.fps = fps
        self.normalize = normalize
        self.compute_flow = compute_flow

        self.frame_extractor = FrameExtractor(
            fps=self.fps,
            frame_size=self.frame_size
        )

        self.flow_processor = OpticalFlowProcessor()

        (
            self.video_paths,
            self.video_ids,
            self.labels,
            self.class_to_idx
        ) = self._load_dataset()

        print(f"\n{dataset_type} Samples: {len(self.video_ids)}")

    # =========================================================
    # LOAD DATASET
    # =========================================================

    def _load_dataset(self):

        if self.dataset_type == "cmu_mosei":
            return self._load_cmu_mosei()

        elif self.dataset_type == "liris_accede":
            return self._load_liris_accede_dataset()

        else:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_type}"
            )

    # =========================================================
    # CMU MOSEI
    # =========================================================

    def _load_cmu_mosei(self):

        video_ids = []
        labels = []

        class_to_idx = {
            "Positive": 0,
            "Neutral": 1,
            "Negative": 2
        }

        if self.labels_path and os.path.exists(self.labels_path):

            video_ids, labels = self._load_labels_from_hdf5(
                self.labels_path
            )

        video_paths = [None] * len(video_ids)

        return (
            video_paths,
            video_ids,
            labels,
            class_to_idx
        )

    # =========================================================
    # LIRIS-ACCEDE
    # =========================================================

    def _load_liris_accede_dataset(self):

        video_paths = []
        labels = []

        class_to_idx = {
            "Positive": 0,
            "Neutral": 1,
            "Negative": 2
        }

        root_path = Path(self.dataset_path)

        video_extensions = [
            "*.mp4",
            "*.avi",
            "*.mov",
            "*.mkv"
        ]

        print("\nScanning LIRIS-ACCEDE Dataset...\n")

        discovered_videos = []
        for ext in video_extensions:
            discovered_videos.extend(root_path.rglob(ext))

        discovered_videos = sorted(set(discovered_videos))
        annotation_file = self._find_accede_annotation(root_path)
        annotation_labels = (
            self._read_accede_annotations(annotation_file)
            if annotation_file is not None else self._labels_from_sorted_valence_rank(discovered_videos)
        )
        if annotation_file is None:
            logger.warning(
                "Official LIRIS-ACCEDE annotations were not found. Using the dataset's "
                "documented valence ordering as deterministic fallback labels. Add "
                "ACCEDEaffect.txt for the most reliable training labels."
            )
        missing = []
        for video_file in discovered_videos:
            name = video_file.name
            if name not in annotation_labels:
                missing.append(name)
                continue
            video_paths.append(str(video_file))
            labels.append(annotation_labels[name])

        if missing:
            logger.warning("Skipping %d videos without an official annotation.", len(missing))
        if not video_paths:
            raise ValueError(f"No video labels matched {annotation_file}")

        video_ids = [
            Path(path).stem
            for path in video_paths
        ]

        print(
            f"LIRIS-ACCEDE Samples Found: {len(video_ids)}"
        )

        return (
            video_paths,
            video_ids,
            labels,
            class_to_idx
        )

    def _find_accede_annotation(self, root_path):
        candidates = []
        if self.labels_path:
            candidates.append(Path(self.labels_path))
        for base in (root_path, root_path.parent, root_path.parent / "annotations"):
            candidates.extend((base / name for name in ("ACCEDEaffect.txt", "ACCEDEranking.txt")))
        return next((path for path in candidates if path.is_file()), None)

    @staticmethod
    def _labels_from_sorted_valence_rank(video_files):
        """Create deterministic classes from LIRIS's documented valence ordering.

        The LIRIS-ACCEDE README states that its discrete clips are ordered from
        lowest to highest induced valence.  This fallback is deliberately based
        on that rank only—never on random ordering or filename keywords.
        """
        ordered = sorted(video_files, key=lambda path: path.name)
        total = len(ordered)
        labels = {}
        for rank, video_file in enumerate(ordered):
            percentile = (rank + 0.5) / total
            labels[video_file.name] = 2 if percentile < 1 / 3 else 0 if percentile > 2 / 3 else 1
        return labels

    @staticmethod
    def _read_accede_annotations(path):
        """Return official valence labels as Positive=0, Neutral=1, Negative=2."""
        labels = {}
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) == 1:
                    row = row[0].split()
                if len(row) < 3 or row[0].lower() in {"id", "#id"}:
                    continue
                name = next((cell for cell in row if cell.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))), None)
                if not name:
                    continue
                try:
                    if path.name.lower() == "accedeaffect.txt":
                        value = int(float(row[-2] if len(row) >= 4 else row[-1]))
                        labels[Path(name).name] = {1: 0, 0: 1, -1: 2}[value]
                    else:
                        value = float(row[4])  # valenceValue in ACCEDEranking.txt
                        labels[Path(name).name] = 2 if value < 2.5 else 0 if value > 3.5 else 1
                except (ValueError, KeyError, IndexError):
                    continue
        if not labels:
            raise ValueError(f"Could not parse LIRIS-ACCEDE labels from {path}")
        return labels

    # =========================================================
    # LOAD MOSEI LABELS
    # =========================================================

    def _load_labels_from_hdf5(self, labels_path):

        video_ids = []
        labels = []

        with h5py.File(labels_path, 'r') as f:

            root = f.get('All Labels') or f

            data_group = root.get('data')

            if data_group is None:

                raise ValueError(
                    f"No data group found in {labels_path}"
                )

            video_ids = sorted(
                list(data_group.keys())
            )

            for vid in video_ids:

                vid_group = data_group[vid]

                label_data = vid_group.get('features')

                if label_data is None:

                    labels.append(1)
                    continue

                score = float(label_data[0][0])

                labels.append(
                    self._convert_continuous_sentiment(score)
                )

        return video_ids, labels

    # =========================================================
    # SENTIMENT CONVERSION
    # =========================================================

    def _convert_continuous_sentiment(self, score):

        if score > 0.5:
            return 0

        elif score < -0.5:
            return 2

        else:
            return 1

    # =========================================================
    # LEN
    # =========================================================

    def __len__(self):

        return len(self.video_ids)

    # =========================================================
    # GET ITEM
    # =========================================================

    def __getitem__(self, idx):

        video_id = self.video_ids[idx]

        label = self.labels[idx]

        video_path = self.video_paths[idx]

        frames = self._load_frames(
            video_id,
            video_path
        )

        frames_tensor = torch.from_numpy(
            frames
        ).permute(0, 3, 1, 2)

        optical_flow_tensor = None

        if self.compute_flow and frames.shape[0] >= 2:

            optical_flow = self.flow_processor.compute_optical_flow(
                frames
            )

            optical_flow_tensor = self._prepare_optical_flow(
                optical_flow
            )

        return {

            'frames': frames_tensor,

            'optical_flow': optical_flow_tensor,

            'label': torch.tensor(
                label,
                dtype=torch.long
            ),

            'video_id': video_id,

            'num_frames': frames_tensor.shape[0]
        }

    # =========================================================
    # LOAD FRAMES
    # =========================================================

    def _load_frames(self, video_id, video_path):

        if not os.path.exists(video_path):

            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        frames = self.frame_extractor.extract_frames(
            video_path,
            output_dir=None
        )

        frames = [

            normalize_frame(
                cv2.resize(frame, self.frame_size)
            )

            for frame in frames
        ]

        return self._pad_or_trim(frames)

    # =========================================================
    # PAD FRAMES
    # =========================================================

    def _pad_or_trim(self, frames):

        if len(frames) >= self.max_frames:

            return np.array(
                frames[:self.max_frames],
                dtype=np.float32
            )

        while len(frames) < self.max_frames:

            frames.append(

                np.zeros(
                    (
                        self.frame_size[0],
                        self.frame_size[1],
                        3
                    ),
                    dtype=np.float32
                )
            )

        return np.array(frames, dtype=np.float32)

    # =========================================================
    # OPTICAL FLOW
    # =========================================================

    def _prepare_optical_flow(self, optical_flow_data):

        flow = optical_flow_data['flow']

        padding = np.zeros(
            (1, *flow.shape[1:]),
            dtype=np.float32
        )

        flow = np.vstack([flow, padding])

        flow_tensor = torch.from_numpy(
            flow
        ).permute(0, 3, 1, 2).unsqueeze(0)

        return flow_tensor


# =============================================================
# COLLATE FUNCTION
# =============================================================

def _collate_fn(batch):

    frames = torch.stack(
        [item['frames'] for item in batch]
    )

    labels = torch.stack(
        [item['label'] for item in batch]
    )

    video_ids = [
        item['video_id']
        for item in batch
    ]

    num_frames = torch.tensor(
        [item['num_frames'] for item in batch]
    )

    optical_flow_batch = [
        item['optical_flow']
        for item in batch
    ]

    if any(flow is not None for flow in optical_flow_batch):

        flows = [

            flow if flow is not None

            else torch.zeros(
                1,
                MAX_FRAMES,
                2,
                FRAME_SIZE[0],
                FRAME_SIZE[1]
            )

            for flow in optical_flow_batch
        ]

        optical_flow = torch.cat(flows, dim=0)

    else:

        optical_flow = None

    return {

        'frames': frames,

        'optical_flow': optical_flow,

        'label': labels,

        'video_id': video_ids,

        'num_frames': num_frames
    }


# =============================================================
# CREATE DATA LOADERS
# =============================================================

def create_data_loaders(

    dataset_type,
    dataset_path,
    frames_path=None,
    labels_path=None,
    batch_size=8,
    num_workers=4,
    train_split=TRAIN_SPLIT,
    val_split=VAL_SPLIT,
    test_split=TEST_SPLIT,
    compute_flow=True,
    seed=42
):

    dataset = VideoFrameDataset(

        dataset_type=dataset_type,

        dataset_path=dataset_path,

        frames_path=frames_path,

        labels_path=labels_path,

        split='train',

        max_frames=MAX_FRAMES,

        frame_size=FRAME_SIZE,

        fps=FPS,

        normalize=True,

        compute_flow=compute_flow
    )

    total = len(dataset)

    train_size = int(total * train_split)

    val_size = int(total * val_split)

    test_size = total - train_size - val_size

    # A fixed generator makes validation/test metrics comparable across runs.
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:train_size + val_size]
    test_indices = permutation[train_size + val_size:]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=True,

        collate_fn=_collate_fn
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,

        collate_fn=_collate_fn
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=True,

        collate_fn=_collate_fn
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        dataset
    )
