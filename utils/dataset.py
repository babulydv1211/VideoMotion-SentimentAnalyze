# """
# Multi-dataset PyTorch loaders for SceneMotion-LLM.
# Supports CMU-MOSEI, UCF101, UCF-Crime, and Kinetics400.
# """

# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# import cv2
# import h5py
# import logging
# from pathlib import Path
# import json
# from .config import FRAME_SIZE, MAX_FRAMES, FPS, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
# from .frame_extractor import FrameExtractor, normalize_frame
# from .optical_flow import OpticalFlowProcessor

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class VideoFrameDataset(Dataset):
#     """Generic video frame dataset with optional optical flow support."""

#     def __init__(
#         self,
#         dataset_type,
#         dataset_path,
#         frames_path=None,
#         labels_path=None,
#         split='train',
#         max_frames=MAX_FRAMES,
#         frame_size=FRAME_SIZE,
#         fps=FPS,
#         normalize=True,
#         transform=None,
#         compute_flow=True
#     ):
#         self.dataset_type = dataset_type
#         self.dataset_path = dataset_path
#         self.frames_path = frames_path
#         self.labels_path = labels_path
#         self.split = split
#         self.max_frames = max_frames
#         self.frame_size = frame_size
#         self.fps = fps
#         self.normalize = normalize
#         self.transform = transform
#         self.compute_flow = compute_flow
#         self.frame_extractor = FrameExtractor(fps=self.fps, frame_size=self.frame_size)
#         self.flow_processor = OpticalFlowProcessor()

#         self.video_paths, self.video_ids, self.labels, self.class_to_idx = self._load_dataset()

#         logger.info(
#             f"Loaded dataset type={self.dataset_type} split={self.split} size={len(self.video_ids)}"
#         )

#     def _load_dataset(self):
#         if self.dataset_type == 'cmu_mosei':
#             return self._load_cmu_mosei()
#         if self.dataset_type in ('ucf101', 'ucf'):
#             return self._load_video_folder_dataset()
#         if self.dataset_type == 'kinetics400':
#             return self._load_video_folder_dataset()
#         if self.dataset_type == 'ucf_crime':
#             return self._load_ucf_crime_dataset()
#         raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

#     def _load_cmu_mosei(self):
#         video_ids = []
#         labels = []
#         class_to_idx = {'Positive': 0, 'Neutral': 1, 'Negative': 2}

#         if self.labels_path and os.path.exists(self.labels_path):
#             video_ids, labels = self._load_labels_from_hdf5(self.labels_path)
#         else:
#             default_labels = os.path.join(self.dataset_path, 'labels', 'CMU_MOSEI_Labels.csd')
#             if os.path.exists(default_labels):
#                 video_ids, labels = self._load_labels_from_hdf5(default_labels)

#         if not video_ids and self.frames_path and os.path.exists(self.frames_path):
#             video_ids = sorted([
#                 d for d in os.listdir(self.frames_path)
#                 if os.path.isdir(os.path.join(self.frames_path, d))
#             ])
#             labels = [1] * len(video_ids)
#             video_paths = [None] * len(video_ids)
#         else:
#             video_paths = [None] * len(video_ids)

#         if not labels:
#             labels = [1] * len(video_ids)

#         return video_paths, video_ids, labels, class_to_idx

#     def _load_video_folder_dataset(self):
#         video_paths = []
#         labels = []
#         class_to_idx = {}

#         if self.labels_path and os.path.exists(self.labels_path):
#             class_to_idx, video_paths, labels = self._parse_labels_file(self.labels_path)
#         else:
#             root_path = Path(self.dataset_path)
#             for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
#                 for video_file in root_path.rglob(ext):
#                     # support UCF101-style split directories: /train/ClassName/video.avi
#                     class_name = video_file.parent.name
#                     if class_name.lower() in {'train', 'val', 'test'} and video_file.parent.parent.exists():
#                         class_name = video_file.parent.parent.name
#                     if class_name not in class_to_idx:
#                         class_to_idx[class_name] = len(class_to_idx)
#                     video_paths.append(str(video_file))
#                     labels.append(class_to_idx[class_name])

#         video_ids = [Path(path).stem for path in video_paths]
#         return video_paths, video_ids, labels, class_to_idx

#     def _load_ucf_crime_dataset(self):
#         video_paths = []
#         labels = []
#         class_to_idx = {'Normal': 0, 'Abnormal': 1}

#         if self.labels_path and os.path.exists(self.labels_path):
#             class_to_idx, video_paths, labels = self._parse_labels_file(self.labels_path)
#         else:
#             root_path = Path(self.dataset_path)
#             for category in ['Normal', 'Abnormal']:
#                 category_dir = root_path / category
#                 if category_dir.exists():
#                     for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
#                         for video_file in category_dir.glob(ext):
#                             video_paths.append(str(video_file))
#                             labels.append(class_to_idx[category])

#         video_ids = [Path(path).stem for path in video_paths]
#         return video_paths, video_ids, labels, class_to_idx

#     def _parse_labels_file(self, labels_path):
#         class_to_idx = {}
#         video_paths = []
#         labels = []

#         with open(labels_path, 'r', encoding='utf-8', errors='ignore') as fp:
#             for line in fp:
#                 line = line.strip()
#                 if not line or line.startswith('#'):
#                     continue
#                 if line.lower().startswith(('clip_name', 'class_name', 'video_name')):
#                     continue

#                 if ',' in line:
#                     parts = [part.strip() for part in line.split(',') if part.strip()]
#                 else:
#                     parts = line.split()

#                 if len(parts) == 2:
#                     video_name, label_token = parts
#                     if label_token.isdigit():
#                         label = int(label_token)
#                     else:
#                         if label_token not in class_to_idx:
#                             class_to_idx[label_token] = len(class_to_idx)
#                         label = class_to_idx[label_token]
#                     video_paths.append(self._find_video_file(video_name))
#                     labels.append(label)
#                 elif len(parts) == 3:
#                     first, second, third = parts
#                     if second.startswith('/') or second.endswith(('.mp4', '.avi', '.mov', '.mkv')):
#                         video_name = first
#                         clip_path = second.lstrip('/\\')
#                         class_name = third
#                         if class_name not in class_to_idx:
#                             class_to_idx[class_name] = len(class_to_idx)
#                         video_path = Path(self.dataset_path) / clip_path
#                         video_paths.append(str(video_path))
#                         labels.append(class_to_idx[class_name])
#                     else:
#                         class_name, video_name, label_token = parts
#                         if label_token.isdigit():
#                             label = int(label_token)
#                         else:
#                             if class_name not in class_to_idx:
#                                 class_to_idx[class_name] = len(class_to_idx)
#                             label = class_to_idx[class_name]
#                         video_paths.append(self._find_video_file(video_name, class_name))
#                         labels.append(label)

#         return class_to_idx, video_paths, labels

#     def _find_video_file(self, video_name, class_name=None):
#         if self.frames_path and os.path.exists(self.frames_path):
#             candidate = Path(self.frames_path) / video_name
#             if candidate.exists():
#                 return str(candidate)

#         for ext in ['.mp4', '.avi', '.mov', '.mkv']:
#             candidate = Path(self.dataset_path) / f"{video_name}{ext}"
#             if candidate.exists():
#                 return str(candidate)
#             if class_name:
#                 candidate = Path(self.dataset_path) / class_name / f"{video_name}{ext}"
#                 if candidate.exists():
#                     return str(candidate)

#         return str(Path(self.dataset_path) / f"{video_name}.mp4")

#     def _load_labels_from_hdf5(self, labels_path):
#         video_ids = []
#         labels = []
#         with h5py.File(labels_path, 'r') as f:
#             root = f.get('All Labels') or f
#             data_group = root.get('data')
#             if data_group is None:
#                 raise ValueError(f"No 'data' group found in labels file: {labels_path}")

#             video_ids = sorted(list(data_group.keys()))
#             for vid in video_ids:
#                 vid_group = data_group[vid]
#                 label_data = vid_group.get('features')
#                 if label_data is None:
#                     labels.append(1)
#                     continue
#                 score = float(label_data[0][0])
#                 labels.append(self._convert_continuous_sentiment(score))

#         return video_ids, labels

#     def _convert_continuous_sentiment(self, score):
#         if score > 0.5:
#             return 0
#         if score < -0.5:
#             return 2
#         return 1

#     def __len__(self):
#         return len(self.video_ids)

#     def __getitem__(self, idx):
#         video_id = self.video_ids[idx]
#         label = self.labels[idx]
#         video_path = self.video_paths[idx] if self.video_paths else None

#         frames = self._load_frames(video_id, video_path)
#         frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

#         if self.transform:
#             frames_tensor = self.transform(frames_tensor)

#         optical_flow_tensor = None
#         if self.compute_flow and frames.shape[0] >= 2:
#             optical_flow = self.flow_processor.compute_optical_flow(frames)
#             optical_flow_tensor = self._prepare_optical_flow(optical_flow)

#         return {
#             'frames': frames_tensor,
#             'optical_flow': optical_flow_tensor,
#             'label': torch.tensor(label, dtype=torch.long),
#             'video_id': video_id,
#             'num_frames': frames_tensor.shape[0]
#         }

#     def _load_frames(self, video_id, video_path=None):
#         if self.frames_path and os.path.exists(self.frames_path):
#             frames_dir = Path(self.frames_path) / video_id
#             if frames_dir.exists():
#                 frame_files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in ['.jpg', '.png']])
#                 if frame_files:
#                     frames = []
#                     for frame_file in frame_files[: self.max_frames]:
#                         frame = cv2.imread(str(frame_file))
#                         if frame is None:
#                             continue
#                         frame = cv2.resize(frame, self.frame_size)
#                         if self.normalize:
#                             frame = frame.astype(np.float32) / 255.0
#                         frames.append(frame)
#                     return self._pad_or_trim(frames)

#         if video_path and os.path.exists(video_path):
#             frames = self.frame_extractor.extract_frames(video_path, output_dir=None)
#             frames = [
#                 normalize_frame(cv2.resize(frame, self.frame_size)) if self.normalize else cv2.resize(frame, self.frame_size)
#                 for frame in frames
#             ]
#             return self._pad_or_trim(frames)

#         video_path = self._resolve_video_path(video_id)
#         if os.path.exists(video_path):
#             frames = self.frame_extractor.extract_frames(video_path, output_dir=None)
#             frames = [
#                 normalize_frame(cv2.resize(frame, self.frame_size)) if self.normalize else cv2.resize(frame, self.frame_size)
#                 for frame in frames
#             ]
#             return self._pad_or_trim(frames)

#         raise FileNotFoundError(f"Video or frames not found for ID: {video_id}")

#     def _resolve_video_path(self, video_id):
#         for ext in ['.mp4', '.avi', '.mov', '.mkv']:
#             candidate = Path(self.dataset_path) / f"{video_id}{ext}"
#             if candidate.exists():
#                 return str(candidate)
#         return str(Path(self.dataset_path) / f"{video_id}.mp4")

#     def _pad_or_trim(self, frames):
#         if len(frames) >= self.max_frames:
#             return np.array(frames[: self.max_frames], dtype=np.float32)
#         while len(frames) < self.max_frames:
#             frames.append(np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.float32))
#         return np.array(frames, dtype=np.float32)

#     def _prepare_optical_flow(self, optical_flow_data):
#         flow = optical_flow_data['flow']
#         padding = np.zeros((1, *flow.shape[1:]), dtype=np.float32)
#         flow = np.vstack([flow, padding])
#         flow_tensor = torch.from_numpy(flow).permute(0, 3, 1, 2).unsqueeze(0)
#         return flow_tensor


# def _collate_fn(batch):
#     frames = torch.stack([item['frames'] for item in batch])
#     labels = torch.stack([item['label'] for item in batch])
#     video_ids = [item['video_id'] for item in batch]
#     num_frames = torch.tensor([item['num_frames'] for item in batch])

#     optical_flow_batch = [item['optical_flow'] for item in batch]
#     if any(flow is not None for flow in optical_flow_batch):
#         flows = [flow if flow is not None else torch.zeros(1, MAX_FRAMES, 2, FRAME_SIZE[0], FRAME_SIZE[1]) for flow in optical_flow_batch]
#         optical_flow = torch.cat(flows, dim=0)
#     else:
#         optical_flow = None

#     return {
#         'frames': frames,
#         'optical_flow': optical_flow,
#         'label': labels,
#         'video_id': video_ids,
#         'num_frames': num_frames
#     }


# def create_data_loaders(
#     dataset_type,
#     dataset_path,
#     frames_path=None,
#     labels_path=None,
#     batch_size=8,
#     num_workers=4,
#     train_split=TRAIN_SPLIT,
#     val_split=VAL_SPLIT,
#     test_split=TEST_SPLIT,
#     compute_flow=True
# ):
#     dataset = VideoFrameDataset(
#         dataset_type=dataset_type,
#         dataset_path=dataset_path,
#         frames_path=frames_path,
#         labels_path=labels_path,
#         split='train',
#         max_frames=MAX_FRAMES,
#         frame_size=FRAME_SIZE,
#         fps=FPS,
#         normalize=True,
#         compute_flow=compute_flow
#     )

#     total = len(dataset)
#     if total == 0:
#         raise ValueError(
#             f"No samples found for dataset_type={dataset_type} at path={dataset_path}. "
#             f"Please verify your dataset structure and labels."
#         )
#     train_size = int(total * train_split)
#     val_size = int(total * val_split)
#     test_size = total - train_size - val_size

#     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
#         dataset,
#         [train_size, val_size, test_size]
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         collate_fn=_collate_fn
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         collate_fn=_collate_fn
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         collate_fn=_collate_fn
#     )

#     return train_loader, val_loader, test_loader, dataset


# class DummyMOSEIDataset(Dataset):
#     """Dummy dataset for testing without real video data."""

#     def __init__(self, num_samples=100, max_frames=MAX_FRAMES, frame_size=FRAME_SIZE):
#         self.num_samples = num_samples
#         self.max_frames = max_frames
#         self.frame_size = frame_size

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         frames = torch.randn(self.max_frames, 3, *self.frame_size)
#         label = torch.tensor(np.random.randint(0, 3), dtype=torch.long)
#         return {
#             'frames': frames,
#             'label': label,
#             'video_id': f'video_{idx:05d}',
#             'num_frames': torch.tensor(self.max_frames)
#         }


# def create_dummy_loaders(batch_size=8, num_samples=100, num_workers=0):
#     train_dataset = DummyMOSEIDataset(num_samples=int(num_samples * 0.7))
#     val_dataset = DummyMOSEIDataset(num_samples=int(num_samples * 0.15))
#     test_dataset = DummyMOSEIDataset(num_samples=int(num_samples * 0.15))

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers
#     )

#     return train_loader, val_loader, test_loader


"""
Multi-dataset PyTorch loaders for SceneMotion-LLM
Supports:
- CMU-MOSEI
- UCF101
- UCF-Crime
- Kinetics400
"""

import os
import cv2
import h5py
import torch
import logging
import numpy as np

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

        elif self.dataset_type == "ucf101":
            return self._load_video_folder_dataset()

        elif self.dataset_type == "kinetics400":
            return self._load_video_folder_dataset()

        elif self.dataset_type == "ucf_crime":
            return self._load_ucf_crime_dataset()

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
    # UCF101 + KINETICS
    # =========================================================

    def _load_video_folder_dataset(self):

        video_paths = []
        labels = []

        class_to_idx = {}

        root_path = Path(self.dataset_path)

        video_extensions = [
            "*.mp4",
            "*.avi",
            "*.mov",
            "*.mkv"
        ]

        for ext in video_extensions:

            for video_file in root_path.rglob(ext):

                class_name = video_file.parent.name

                if class_name not in class_to_idx:

                    class_to_idx[class_name] = len(class_to_idx)

                video_paths.append(str(video_file))

                labels.append(
                    class_to_idx[class_name]
                )

        video_ids = [
            Path(path).stem
            for path in video_paths
        ]

        return (
            video_paths,
            video_ids,
            labels,
            class_to_idx
        )

    # =========================================================
    # UCF CRIME
    # =========================================================

    def _load_ucf_crime_dataset(self):

        video_paths = []
        labels = []

        class_to_idx = {
            "Normal": 0,
            "Abnormal": 1
        }

        root_path = Path(self.dataset_path)

        video_extensions = [
            "*.mp4",
            "*.avi",
            "*.mov",
            "*.mkv"
        ]

        print("\nScanning UCF-Crime Dataset...\n")

        for ext in video_extensions:

            for video_file in root_path.rglob(ext):

                video_name = str(video_file).lower()

                # NORMAL
                if "normal" in video_name:

                    label = class_to_idx["Normal"]

                # ABNORMAL
                else:

                    label = class_to_idx["Abnormal"]

                video_paths.append(str(video_file))
                labels.append(label)

        video_ids = [
            Path(path).stem
            for path in video_paths
        ]

        print(
            f"UCF-Crime Samples Found: {len(video_ids)}"
        )

        return (
            video_paths,
            video_ids,
            labels,
            class_to_idx
        )

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
    compute_flow=True
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

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(

        dataset,

        [
            train_size,
            val_size,
            test_size
        ]
    )

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