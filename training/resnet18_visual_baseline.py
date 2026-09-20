"""Offline frozen ResNet-18 features and a compact LIRIS baseline.

This is intentionally a separate experiment path from the semantic
five-pillar model.  It keeps the full 512-dimensional ResNet-18 penultimate
layer for up to sixteen uniformly sampled frames per video, rather than
compressing it through a fixed random projection.  The raw feature cache is
portable, source-fingerprinted, and can be reused for future small heads.

The training baseline is deliberately conservative:

* PCA is fit on *training-frame embeddings only*.
* Per-video mean, standard deviation, and start-to-end change are pooled from
  the reduced frame sequence.
* A class-balanced multinomial logistic regression is selected only on the
  official validation split, then evaluated once on the held-out test split.

No code in this module downloads a model.  It refuses to start extraction
unless torchvision's locally cached ``ResNet18_Weights.IMAGENET1K_V1`` file is
already present.  This makes the command suitable for an offline workstation.

Example::

    python -m scene_motion_llm.training.resnet18_visual_baseline precompute \
      --video-dir .\\scene_motion_llm\\dataset\\Liris_Accede \
      --ranking-path .\\scene_motion_llm\\dataset\\annotations\\ACCEDEranking.txt \
      --sets-path C:\\path\\to\\ACCEDEsets.txt \
      --cache-dir .\\scene_motion_llm\\cache\\resnet18_full_v1 --device cuda

    python -m scene_motion_llm.training.resnet18_visual_baseline train \
      --video-dir .\\scene_motion_llm\\dataset\\Liris_Accede \
      --ranking-path .\\scene_motion_llm\\dataset\\annotations\\ACCEDEranking.txt \
      --sets-path C:\\path\\to\\ACCEDEsets.txt \
      --cache-dir .\\scene_motion_llm\\cache\\resnet18_full_v1 \
      --output-dir .\\scene_motion_llm\\checkpoints\\resnet18_pca_logreg
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid

import numpy as np


RESNET18_VISUAL_CACHE_VERSION = "resnet18-visual-cache-v1"
"""Version of the raw, full-embedding cache format."""

RESNET18_EMBEDDING_DIM = 512
RESNET18_WEIGHTS_ID = "ResNet18_Weights.IMAGENET1K_V1"
CLASS_NAMES = ("Positive", "Neutral", "Negative")
OFFICIAL_SPLITS = ("train", "validation", "test")
BASELINE_TASK = "liris_resnet18_pca_logistic_regression"


@dataclass(frozen=True)
class ResNet18FeatureConfig:
    """Immutable preprocessing contract for the raw visual feature cache.

    Videos are bounded to the first ``max_duration_seconds`` before sampling.
    That is compatible with the existing short-video protocol and avoids an
    accidental unbounded decode if a malformed source file is encountered.
    Standard LIRIS clips yield exactly sixteen embeddings; genuinely shorter
    clips yield one embedding per available frame rather than fabricated
    duplicates.
    """

    sampled_frames: int = 16
    max_duration_seconds: float = 15.0
    resize_shorter_side: int = 256
    crop_size: int = 224
    inference_batch_size: int = 16

    def __post_init__(self) -> None:
        if self.sampled_frames < 1:
            raise ValueError("sampled_frames must be at least 1")
        if not math.isfinite(float(self.max_duration_seconds)) or self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be a finite positive number")
        if self.resize_shorter_side < self.crop_size:
            raise ValueError("resize_shorter_side must be at least crop_size")
        if self.crop_size < 1:
            raise ValueError("crop_size must be at least 1")
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be at least 1")

    def metadata(self) -> dict[str, Any]:
        """Return the complete, JSON-safe preprocessing contract."""

        return {
            "cache_version": RESNET18_VISUAL_CACHE_VERSION,
            "weights": RESNET18_WEIGHTS_ID,
            "embedding_dim": RESNET18_EMBEDDING_DIM,
            "sampling": "uniform_frame_indices_within_first_max_duration_seconds",
            "preprocessing": "resize_shorter_side_then_center_crop_then_imagenet_normalize",
            "config": asdict(self),
        }

    def cache_signature(self) -> str:
        """Stable signature used to reject incompatible/stale feature files."""

        encoded = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]


@dataclass(frozen=True)
class ResNet18Record:
    """Small record type accepted by cache/training helpers.

    ``LirisRecord`` from ``semantic_five_pillar_training`` has the same four
    attributes, so callers can pass canonical official-split records directly
    without converting them.
    """

    video_id: str
    video_path: str
    label: int
    split: str


@dataclass(frozen=True)
class ResNet18FeatureResult:
    """Full frozen visual sequence for one raw video."""

    embeddings: np.ndarray
    frame_indices: np.ndarray
    timestamps_seconds: np.ndarray
    video_metadata: Mapping[str, Any]

    def validated(self) -> "ResNet18FeatureResult":
        """Return canonical float32/int64 arrays after enforcing the contract."""

        embeddings = np.asarray(self.embeddings, dtype=np.float32)
        indices = np.asarray(self.frame_indices, dtype=np.int64)
        timestamps = np.asarray(self.timestamps_seconds, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != RESNET18_EMBEDDING_DIM:
            raise ValueError(
                "ResNet-18 embeddings must have shape (frames, "
                f"{RESNET18_EMBEDDING_DIM}), got {embeddings.shape}."
            )
        if embeddings.shape[0] < 1:
            raise ValueError("A visual feature sequence must contain at least one frame.")
        if indices.ndim != 1 or timestamps.ndim != 1:
            raise ValueError("frame_indices and timestamps_seconds must be one-dimensional.")
        if len(indices) != embeddings.shape[0] or len(timestamps) != embeddings.shape[0]:
            raise ValueError("Feature arrays must have the same sequence length.")
        if not np.isfinite(embeddings).all() or not np.isfinite(timestamps).all():
            raise ValueError("ResNet-18 feature arrays must be finite.")
        if (indices < 0).any() or (np.diff(indices) < 0).any():
            raise ValueError("frame_indices must be non-negative and non-decreasing.")
        return ResNet18FeatureResult(
            embeddings=np.ascontiguousarray(embeddings),
            frame_indices=np.ascontiguousarray(indices),
            timestamps_seconds=np.ascontiguousarray(timestamps),
            video_metadata=dict(self.video_metadata),
        )


def _json_safe(value: Any) -> Any:
    """Convert small NumPy/path values to ordinary JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist JSON so an interrupted run never looks complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(_json_safe(payload), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_npz_dump(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Atomically persist compressed arrays in the destination cache folder."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez_compressed(str(temporary), **payload)
        os.replace(temporary, path)
        temporary = None  # type: ignore[assignment]
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _source_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _safe_cache_stem(video_id: str) -> str:
    stem = "".join(character if character.isalnum() or character in "-_." else "_" for character in str(video_id))
    if not stem or stem in {".", ".."}:
        raise ValueError(f"Unsafe video id for cache filename: {video_id!r}")
    return stem


def _record_field(record: Any, name: str) -> Any:
    try:
        return getattr(record, name)
    except AttributeError as error:
        raise TypeError(f"Record {record!r} is missing required field {name!r}.") from error


def uniform_frame_indices(frame_count: int, sampled_frames: int) -> np.ndarray:
    """Return ordered, non-duplicated uniform indices for a bounded video."""

    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if sampled_frames < 1:
        raise ValueError("sampled_frames must be at least 1")
    count = min(int(frame_count), int(sampled_frames))
    indices = np.linspace(0, frame_count - 1, num=count, dtype=np.int64)
    if len(np.unique(indices)) != len(indices):  # Defensive for future sampling changes.
        raise RuntimeError("Uniform frame sampling unexpectedly produced duplicate indices.")
    return indices


def _decode_uniform_frames(
    video_path: str | Path,
    config: ResNet18FeatureConfig,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode at most one bounded source segment and sample frames uniformly.

    ``cv2.VideoCapture`` usually exposes frame count.  If a container does not,
    this function performs one bounded sequential decode rather than retaining
    an unlimited video in memory.
    """

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - dependency is pinned by the project.
        raise RuntimeError("OpenCV is required to decode videos for ResNet-18 features.") from error

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"Video file was not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not decode video: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = max(1, int(math.floor(config.max_duration_seconds * fps)))
        source_duration_seconds: float | None = None
        if reported_frame_count > 0:
            source_duration_seconds = reported_frame_count / fps
            usable_frame_count = min(reported_frame_count, max_frames)
            indices = uniform_frame_indices(usable_frame_count, config.sampled_frames)
            frames: list[np.ndarray] = []
            for index in indices.tolist():
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"Unable to seek sampled frame {index} in {source}")
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(f"Sampled frame {index} in {source} is not BGR colour data")
                frames.append(frame)
            timestamps = indices.astype(np.float32) / np.float32(fps)
            metadata = {
                "fps": float(fps),
                "reported_frame_count": int(reported_frame_count),
                "bounded_frame_count": int(usable_frame_count),
                "source_duration_seconds": float(source_duration_seconds),
                "truncated_to_max_duration": bool(reported_frame_count > max_frames),
                "frame_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "frame_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            return frames, indices, timestamps, metadata

        # Some damaged containers report no frame count.  Retain at most the
        # configured duration, then sample from that bounded decoded sequence.
        decoded: list[np.ndarray] = []
        while len(decoded) < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"A decoded frame in {source} is not BGR colour data")
            decoded.append(frame)
        if not decoded:
            raise ValueError(f"No readable frames found in: {source}")
        indices = uniform_frame_indices(len(decoded), config.sampled_frames)
        frames = [decoded[int(index)] for index in indices]
        timestamps = indices.astype(np.float32) / np.float32(fps)
        height, width = decoded[0].shape[:2]
        metadata = {
            "fps": float(fps),
            "reported_frame_count": 0,
            "bounded_frame_count": int(len(decoded)),
            "source_duration_seconds": None,
            "truncated_to_max_duration": bool(len(decoded) == max_frames),
            "frame_width": int(width),
            "frame_height": int(height),
        }
        return frames, indices, timestamps, metadata
    finally:
        capture.release()


def _resolve_device(device: str | None) -> Any:
    """Resolve ``auto`` while rejecting an unavailable explicitly requested GPU."""

    import torch

    requested = (device or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for ResNet-18 extraction but no CUDA device is available.")
    return resolved


def _local_resnet18_checkpoint() -> tuple[Any, Path, Any]:
    """Return explicit ImageNet-V1 weights only when their file is local.

    The file existence check happens before ``torchvision`` constructs the
    model, so this code path never turns a missing cache into a network
    request.
    """

    import torch
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1
    checkpoint_name = Path(str(weights.url)).name
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Frozen ResNet-18 extraction is offline-only, but the required "
            f"local checkpoint is absent: {checkpoint}. Expected {checkpoint_name}. "
            "No model download was attempted."
        )
    return weights, checkpoint, torch


class FrozenResNet18Extractor:
    """Load one local frozen ResNet-18 and extract full per-frame embeddings."""

    def __init__(
        self,
        config: ResNet18FeatureConfig | None = None,
        *,
        device: str | None = "auto",
    ) -> None:
        self.config = config or ResNet18FeatureConfig()
        self.device = _resolve_device(device)
        weights, checkpoint, torch = _local_resnet18_checkpoint()
        from torchvision.models import resnet18

        # The preceding explicit cache check makes this safe in offline mode.
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model = model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._torch = torch
        self._model = model
        self._normalization_mean = torch.tensor(
            (0.485, 0.456, 0.406), dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self._normalization_std = torch.tensor(
            (0.229, 0.224, 0.225), dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        checkpoint_stat = checkpoint.stat()
        self.model_metadata: dict[str, Any] = {
            "weights": RESNET18_WEIGHTS_ID,
            "checkpoint_name": checkpoint.name,
            "checkpoint_size_bytes": int(checkpoint_stat.st_size),
            "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
            "device": str(self.device),
            "embedding_layer": "post_avgpool_pre_classifier",
        }

    def _preprocess(self, frames: Sequence[np.ndarray]) -> Any:
        """Apply deterministic ImageNet resize/crop/normalization to a batch."""

        if not frames:
            raise ValueError("Cannot preprocess an empty frame batch.")
        rgb = np.stack([frame[..., ::-1] for frame in frames], axis=0).copy()
        tensor = self._torch.from_numpy(rgb).permute(0, 3, 1, 2).float().div_(255.0)
        tensor = tensor.to(self.device, non_blocking=self.device.type == "cuda")
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        if height < 1 or width < 1:
            raise ValueError("A sampled frame has invalid dimensions.")
        scale = self.config.resize_shorter_side / float(min(height, width))
        target_height = max(self.config.crop_size, int(round(height * scale)))
        target_width = max(self.config.crop_size, int(round(width * scale)))
        tensor = self._torch.nn.functional.interpolate(
            tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        top = (target_height - self.config.crop_size) // 2
        left = (target_width - self.config.crop_size) // 2
        tensor = tensor[:, :, top : top + self.config.crop_size, left : left + self.config.crop_size]
        return (tensor - self._normalization_mean) / self._normalization_std

    def extract_video(self, video_path: str | Path) -> ResNet18FeatureResult:
        """Extract full 512-D embeddings from uniformly sampled raw frames."""

        frames, indices, timestamps, video_metadata = _decode_uniform_frames(video_path, self.config)
        pieces: list[np.ndarray] = []
        for start in range(0, len(frames), self.config.inference_batch_size):
            batch = self._preprocess(frames[start : start + self.config.inference_batch_size])
            with self._torch.inference_mode():
                output = self._model(batch)
            if output.ndim != 2 or output.shape[1] != RESNET18_EMBEDDING_DIM:
                raise RuntimeError(
                    "Frozen ResNet-18 returned an unexpected embedding shape: "
                    f"{tuple(output.shape)}."
                )
            pieces.append(output.detach().float().cpu().numpy())
        result = ResNet18FeatureResult(
            embeddings=np.concatenate(pieces, axis=0),
            frame_indices=indices,
            timestamps_seconds=timestamps,
            video_metadata=video_metadata,
        )
        return result.validated()


class ResNet18FeatureCache:
    """Versioned full-embedding cache with source and config validation."""

    def __init__(self, cache_dir: str | Path, config: ResNet18FeatureConfig | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config or ResNet18FeatureConfig()

    def paths_for(self, record: Any) -> tuple[Path, Path]:
        split = str(_record_field(record, "split"))
        if split not in OFFICIAL_SPLITS:
            raise ValueError(f"Unknown LIRIS split for ResNet-18 cache: {split!r}")
        stem = _safe_cache_stem(str(_record_field(record, "video_id")))
        directory = self.cache_dir / split
        return directory / f"{stem}.npz", directory / f"{stem}.json"

    def load(self, record: Any) -> ResNet18FeatureResult | None:
        """Return a feature sequence only when its complete cache contract matches."""

        array_path, metadata_path = self.paths_for(record)
        metadata = _read_json(metadata_path)
        if metadata is None or not array_path.is_file():
            return None
        expected_fingerprint = _source_fingerprint(str(_record_field(record, "video_path")))
        expected = {
            "cache_version": RESNET18_VISUAL_CACHE_VERSION,
            "cache_signature": self.config.cache_signature(),
            "video_id": str(_record_field(record, "video_id")),
            "split": str(_record_field(record, "split")),
            "source_fingerprint": expected_fingerprint,
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            return None
        try:
            with np.load(array_path, allow_pickle=False) as arrays:
                result = ResNet18FeatureResult(
                    embeddings=np.asarray(arrays["embeddings"], dtype=np.float32),
                    frame_indices=np.asarray(arrays["frame_indices"], dtype=np.int64),
                    timestamps_seconds=np.asarray(arrays["timestamps_seconds"], dtype=np.float32),
                    video_metadata=metadata.get("video_metadata", {}),
                ).validated()
            if int(metadata.get("sequence_length", -1)) != result.embeddings.shape[0]:
                return None
            if int(metadata.get("embedding_dim", -1)) != RESNET18_EMBEDDING_DIM:
                return None
            return result
        except (OSError, KeyError, ValueError):
            return None

    def store(
        self,
        record: Any,
        result: ResNet18FeatureResult,
        *,
        model_metadata: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        """Write a complete array/metadata pair atomically."""

        valid = result.validated()
        array_path, metadata_path = self.paths_for(record)
        _atomic_npz_dump(
            array_path,
            {
                "embeddings": valid.embeddings,
                "frame_indices": valid.frame_indices,
                "timestamps_seconds": valid.timestamps_seconds,
            },
        )
        metadata = {
            "cache_version": RESNET18_VISUAL_CACHE_VERSION,
            "cache_signature": self.config.cache_signature(),
            "feature_config": self.config.metadata(),
            "video_id": str(_record_field(record, "video_id")),
            "split": str(_record_field(record, "split")),
            "source_fingerprint": _source_fingerprint(str(_record_field(record, "video_path"))),
            "model": dict(model_metadata),
            "sequence_length": int(valid.embeddings.shape[0]),
            "embedding_dim": RESNET18_EMBEDDING_DIM,
            "video_metadata": dict(valid.video_metadata),
        }
        _atomic_json_dump(metadata_path, metadata)
        return array_path, metadata_path


def precompute_resnet18_features(
    records_by_split: Mapping[str, Sequence[Any]],
    cache_dir: str | Path,
    *,
    config: ResNet18FeatureConfig | None = None,
    device: str | None = "auto",
    limit_per_split: int | None = None,
    force: bool = False,
    fail_fast: bool = False,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Precompute/reuse full ResNet-18 cache entries for official records.

    The function intentionally uses one process and one frozen model instance.
    This is usually faster and more stable on a single GPU than spawning model
    workers, and it avoids loading the 47 MB checkpoint once per video.
    Failures are reported explicitly and never become zero-valued features.
    """

    active_config = config or ResNet18FeatureConfig()
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError("limit_per_split must be at least 1 when supplied")
    cache = ResNet18FeatureCache(cache_dir, active_config)
    extractor: FrozenResNet18Extractor | None = None
    report: dict[str, Any] = {
        "task": "liris_resnet18_full_embedding_precompute",
        "cache_dir": str(Path(cache_dir)),
        "cache_signature": active_config.cache_signature(),
        "feature_config": active_config.metadata(),
        "splits": {},
        "failures": [],
    }
    for split in OFFICIAL_SPLITS:
        records = list(records_by_split.get(split, ()))
        if not records:
            raise ValueError(f"No records provided for required official split {split!r}.")
        if limit_per_split is not None:
            records = records[:limit_per_split]
        counts = {"requested": len(records), "cached": 0, "extracted": 0, "failed": 0}
        for index, record in enumerate(records, start=1):
            if str(_record_field(record, "split")) != split:
                raise ValueError(
                    f"Record {_record_field(record, 'video_id')!r} is in mapping key {split!r} "
                    f"but declares split {_record_field(record, 'split')!r}."
                )
            try:
                cached = None if force else cache.load(record)
                if cached is not None:
                    counts["cached"] += 1
                    status = "cached"
                else:
                    if extractor is None:
                        extractor = FrozenResNet18Extractor(active_config, device=device)
                        report["model"] = dict(extractor.model_metadata)
                    result = extractor.extract_video(str(_record_field(record, "video_path")))
                    cache.store(record, result, model_metadata=extractor.model_metadata)
                    counts["extracted"] += 1
                    status = "extracted"
                if progress_callback is not None:
                    progress_callback(
                        {
                            "split": split,
                            "index": index,
                            "total": len(records),
                            "video_id": str(_record_field(record, "video_id")),
                            "status": status,
                        }
                    )
            except Exception as error:
                counts["failed"] += 1
                failure = {
                    "split": split,
                    "video_id": str(_record_field(record, "video_id")),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                report["failures"].append(failure)
                if progress_callback is not None:
                    progress_callback({**failure, "index": index, "total": len(records), "status": "failed"})
                if fail_fast:
                    raise RuntimeError(f"ResNet-18 cache extraction failed: {failure}") from error
        report["splits"][split] = counts
    report["model"] = report.get("model", "not_loaded_all_entries_were_valid_cache_hits")
    return report


def _load_cached_sequences(
    records: Sequence[Any], cache: ResNet18FeatureCache, *, split: str
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Load complete, fingerprint-valid sequences or explain exactly what is missing."""

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    missing: list[str] = []
    for record in records:
        if str(_record_field(record, "split")) != split:
            raise ValueError(f"Record {_record_field(record, 'video_id')!r} has an unexpected split.")
        result = cache.load(record)
        if result is None:
            missing.append(str(_record_field(record, "video_id")))
            continue
        label = int(_record_field(record, "label"))
        if label < 0 or label >= len(CLASS_NAMES):
            raise ValueError(f"Invalid class label {label} for {_record_field(record, 'video_id')!r}.")
        sequences.append(result.embeddings)
        labels.append(label)
        ids.append(str(_record_field(record, "video_id")))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} {split} ResNet-18 feature cache entries are missing, stale, or invalid; "
            f"examples: {missing[:5]}. Run the precompute command before training."
        )
    if not sequences:
        raise ValueError(f"No valid cached feature sequences were found for {split!r}.")
    return sequences, np.asarray(labels, dtype=np.int64), ids


def fit_train_only_pca(
    training_sequences: Sequence[np.ndarray], *, pca_components: int = 96, seed: int = 20_260_917
) -> Any:
    """Fit PCA on training-frame rows only; never pass validation/test frames here."""

    if pca_components < 1:
        raise ValueError("pca_components must be at least 1")
    if not training_sequences:
        raise ValueError("At least one training sequence is required for PCA.")
    matrix = np.concatenate([np.asarray(sequence, dtype=np.float32) for sequence in training_sequences], axis=0)
    if matrix.ndim != 2 or matrix.shape[1] != RESNET18_EMBEDDING_DIM:
        raise ValueError(f"Training frame matrix must have shape (N, {RESNET18_EMBEDDING_DIM}).")
    if not np.isfinite(matrix).all():
        raise ValueError("Training frame embeddings must be finite.")
    usable_components = min(int(pca_components), int(matrix.shape[0]), int(matrix.shape[1]))
    from sklearn.decomposition import PCA

    reducer = PCA(n_components=usable_components, svd_solver="randomized", random_state=int(seed))
    reducer.fit(matrix)
    return reducer


def pool_reduced_sequence(reduced_frames: np.ndarray) -> np.ndarray:
    """Compactly summarize visual appearance, variability, and temporal change."""

    frames = np.asarray(reduced_frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[0] < 1:
        raise ValueError("Reduced frame embeddings must have shape (frames, components).")
    if not np.isfinite(frames).all():
        raise ValueError("Reduced frame embeddings must be finite.")
    mean = frames.mean(axis=0)
    std = frames.std(axis=0)
    delta = frames[-1] - frames[0] if frames.shape[0] > 1 else np.zeros(frames.shape[1], dtype=np.float32)
    return np.ascontiguousarray(np.concatenate((mean, std, delta)).astype(np.float32, copy=False))


def transform_and_pool_sequences(reducer: Any, sequences: Sequence[np.ndarray]) -> np.ndarray:
    """Apply a fitted PCA and form one compact vector per video."""

    if not hasattr(reducer, "transform"):
        raise TypeError("reducer must be a fitted sklearn-style transformer.")
    vectors = [pool_reduced_sequence(np.asarray(reducer.transform(sequence), dtype=np.float32)) for sequence in sequences]
    if not vectors:
        raise ValueError("At least one sequence is required to build pooled vectors.")
    return np.stack(vectors).astype(np.float32, copy=False)


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    """Return accuracy, balanced accuracy, macro-F1, and transparent per-class metrics."""

    if len(labels) != len(predictions) or not labels:
        raise ValueError("labels and predictions must be non-empty equal-length sequences.")
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for target, predicted in zip(labels, predictions, strict=True):
        if not 0 <= int(target) < len(CLASS_NAMES) or not 0 <= int(predicted) < len(CLASS_NAMES):
            raise ValueError(f"Invalid class pair target={target}, prediction={predicted}.")
        confusion[int(target), int(predicted)] += 1
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = np.divide(true_positive, predicted_count, out=np.zeros_like(true_positive), where=predicted_count > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    valid_recall = recall[support > 0]
    return {
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "balanced_accuracy": float(valid_recall.mean()) if valid_recall.size else float("nan"),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(CLASS_NAMES))
        },
        "support": int(confusion.sum()),
    }


def _atomic_joblib_dump(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write the fitted sklearn objects and their protocol metadata."""

    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.joblib"
    try:
        joblib.dump(dict(payload), temporary, compress=3)
        os.replace(temporary, path)
        temporary = None  # type: ignore[assignment]
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _validate_c_values(c_values: Iterable[float]) -> tuple[float, ...]:
    values = tuple(sorted({float(value) for value in c_values}))
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("C values must be a non-empty collection of finite positive numbers.")
    return values


def train_resnet18_pca_logistic_baseline(
    records_by_split: Mapping[str, Sequence[Any]],
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    feature_config: ResNet18FeatureConfig | None = None,
    pca_components: int = 96,
    c_values: Iterable[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    seed: int = 20_260_917,
    max_iter: int = 2_000,
) -> dict[str, Any]:
    """Fit a validation-selected, compact visual baseline from a valid cache.

    The held-out test features are deliberately loaded only after validation
    chooses ``C``.  PCA and feature standardization are fit solely on training
    videos, and the final selected classifier remains train-only so the test
    score stays a proper unseen-movie measurement.
    """

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    active_config = feature_config or ResNet18FeatureConfig()
    candidate_c_values = _validate_c_values(c_values)
    for split in OFFICIAL_SPLITS:
        if not records_by_split.get(split):
            raise ValueError(f"No records supplied for required official split {split!r}.")
    cache = ResNet18FeatureCache(cache_dir, active_config)
    train_sequences, train_labels, train_ids = _load_cached_sequences(records_by_split["train"], cache, split="train")
    validation_sequences, validation_labels, validation_ids = _load_cached_sequences(
        records_by_split["validation"], cache, split="validation"
    )
    reducer = fit_train_only_pca(train_sequences, pca_components=pca_components, seed=seed)
    train_vectors = transform_and_pool_sequences(reducer, train_sequences)
    validation_vectors = transform_and_pool_sequences(reducer, validation_sequences)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_vectors)
    scaled_train = scaler.transform(train_vectors)
    scaled_validation = scaler.transform(validation_vectors)
    candidates: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float], float, Any, dict[str, Any]] | None = None
    for c_value in candidate_c_values:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=int(max_iter),
            random_state=int(seed),
        )
        classifier.fit(scaled_train, train_labels)
        metrics = classification_metrics(validation_labels.tolist(), classifier.predict(scaled_validation).tolist())
        candidates.append({"C": c_value, "validation": metrics})
        # Higher macro-F1 is primary, then balanced accuracy, then accuracy.
        # Iterating C in ascending order makes an exact tie choose the more
        # regularized model rather than a larger, less stable coefficient set.
        score = (metrics["macro_f1"], metrics["balanced_accuracy"], metrics["accuracy"])
        if best is None or score > best[0]:
            best = (score, c_value, classifier, metrics)
    assert best is not None  # candidate C validation above guarantees this.
    _, selected_c, selected_classifier, selected_validation_metrics = best

    # Test is intentionally untouched until model selection is complete.
    test_sequences, test_labels, test_ids = _load_cached_sequences(records_by_split["test"], cache, split="test")
    test_vectors = transform_and_pool_sequences(reducer, test_sequences)
    test_predictions = selected_classifier.predict(scaler.transform(test_vectors))
    test_metrics = classification_metrics(test_labels.tolist(), test_predictions.tolist())
    train_metrics = classification_metrics(train_labels.tolist(), selected_classifier.predict(scaled_train).tolist())

    target_dir = Path(output_dir)
    checkpoint_path = target_dir / "resnet18_pca_logistic_baseline.joblib"
    report_path = target_dir / "resnet18_pca_logistic_baseline.json"
    protocol = {
        "pca_fit_split": "train_frames_only",
        "scaler_fit_split": "train_videos_only",
        "classifier_fit_split": "train_videos_only",
        "selection_split": "validation",
        "test_usage": "single_evaluation_after_validation_selection",
        "pooling": "per_component_mean_std_start_to_end_delta",
        "class_weight": "balanced",
    }
    model_payload = {
        "task": BASELINE_TASK,
        "class_names": list(CLASS_NAMES),
        "feature_config": active_config.metadata(),
        "cache_signature": active_config.cache_signature(),
        "pca": reducer,
        "scaler": scaler,
        "classifier": selected_classifier,
        "selected_c": float(selected_c),
        "seed": int(seed),
        "protocol": protocol,
    }
    _atomic_joblib_dump(checkpoint_path, model_payload)
    report = {
        "task": BASELINE_TASK,
        "checkpoint": str(checkpoint_path),
        "feature_config": active_config.metadata(),
        "cache_signature": active_config.cache_signature(),
        "class_names": list(CLASS_NAMES),
        "seed": int(seed),
        "pca_components_requested": int(pca_components),
        "pca_components_fitted": int(reducer.n_components_),
        "pooled_feature_dim": int(train_vectors.shape[1]),
        "selected_c": float(selected_c),
        "candidates": candidates,
        "protocol": protocol,
        "split_sizes": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "metrics": {
            "train": train_metrics,
            "validation": selected_validation_metrics,
            "test": test_metrics,
        },
    }
    _atomic_json_dump(report_path, report)
    report["report"] = str(report_path)
    return report


def _build_records_from_cli(args: argparse.Namespace) -> Mapping[str, Sequence[Any]]:
    """Use the canonical official LIRIS join without coupling core helpers to it."""

    from scene_motion_llm.training.semantic_five_pillar_training import build_liris_records

    return build_liris_records(
        args.video_dir,
        args.ranking_path,
        args.sets_path,
        require_complete=not bool(getattr(args, "allow_partial", False)),
    )


def _parse_c_values(value: str) -> tuple[float, ...]:
    try:
        return _validate_c_values(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _feature_config_from_args(args: argparse.Namespace) -> ResNet18FeatureConfig:
    return ResNet18FeatureConfig(
        sampled_frames=int(args.sampled_frames),
        max_duration_seconds=float(args.max_duration_seconds),
        resize_shorter_side=int(args.resize_shorter_side),
        crop_size=int(args.crop_size),
        inference_batch_size=int(args.inference_batch_size),
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video-dir", required=True, help="Directory containing raw LIRIS .mp4 clips.")
    parser.add_argument("--ranking-path", required=True, help="Official ACCEDEranking.txt path.")
    parser.add_argument("--sets-path", required=True, help="Official ACCEDEsets.txt path.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a deliberately partial video directory (smoke tests only).",
    )


def _add_feature_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sampled-frames", type=int, default=16, help="Uniform ResNet frames per clip (default: 16).")
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=15.0,
        help="Bound extraction to this leading source duration (default: 15).",
    )
    parser.add_argument("--resize-shorter-side", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--inference-batch-size", type=int, default=16)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the offline cache/training command-line interface."""

    parser = argparse.ArgumentParser(
        description="Offline frozen ResNet-18 visual LIRIS baseline; it never downloads model weights."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    precompute = subcommands.add_parser("precompute", help="Cache full 512-D frame embeddings.")
    _add_dataset_arguments(precompute)
    _add_feature_arguments(precompute)
    precompute.add_argument("--cache-dir", required=True, help="Destination for versioned .npz/.json feature pairs.")
    precompute.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu (default: auto).")
    precompute.add_argument("--limit-per-split", type=int, help="Small extraction smoke-test limit.")
    precompute.add_argument("--force", action="store_true", help="Re-extract even a currently valid cache entry.")
    precompute.add_argument("--fail-fast", action="store_true", help="Stop at the first failed source video.")
    precompute.add_argument("--progress-every", type=int, default=25, help="Print a progress JSON line every N clips.")
    precompute.add_argument("--report-path", help="Optional explicit precompute report JSON path.")

    train = subcommands.add_parser("train", help="Fit train-only PCA plus validation-selected logistic regression.")
    _add_dataset_arguments(train)
    _add_feature_arguments(train)
    train.add_argument("--cache-dir", required=True, help="Existing valid full-embedding feature cache.")
    train.add_argument("--output-dir", required=True, help="Destination for model checkpoint and JSON report.")
    train.add_argument("--pca-components", type=int, default=96, help="Maximum train-only PCA components (default: 96).")
    train.add_argument(
        "--c-values",
        type=_parse_c_values,
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
        help="Comma-separated logistic C candidates; selected on validation macro-F1.",
    )
    train.add_argument("--seed", type=int, default=20_260_917)
    train.add_argument("--max-iter", type=int, default=2_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a cache or compact-classifier experiment and print a JSON report."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    records = _build_records_from_cli(args)
    feature_config = _feature_config_from_args(args)
    if args.command == "precompute":
        if args.progress_every < 1:
            parser.error("--progress-every must be at least 1")
        seen = 0

        def progress(event: Mapping[str, Any]) -> None:
            nonlocal seen
            seen += 1
            if seen % int(args.progress_every) == 0 or event.get("status") == "failed":
                print(json.dumps(_json_safe(event), sort_keys=True), flush=True)

        report = precompute_resnet18_features(
            records,
            args.cache_dir,
            config=feature_config,
            device=args.device,
            limit_per_split=args.limit_per_split,
            force=bool(args.force),
            fail_fast=bool(args.fail_fast),
            progress_callback=progress,
        )
        report_path = Path(args.report_path) if args.report_path else Path(args.cache_dir) / "precompute_resnet18_visual_report.json"
        _atomic_json_dump(report_path, report)
        report["report"] = str(report_path)
    elif args.command == "train":
        report = train_resnet18_pca_logistic_baseline(
            records,
            args.cache_dir,
            args.output_dir,
            feature_config=feature_config,
            pca_components=args.pca_components,
            c_values=args.c_values,
            seed=args.seed,
            max_iter=args.max_iter,
        )
    else:  # pragma: no cover - argparse enforces the subcommands.
        raise AssertionError(f"Unknown command: {args.command!r}")
    print(json.dumps(_json_safe(report), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    raise SystemExit(main())


__all__ = [
    "BASELINE_TASK",
    "CLASS_NAMES",
    "FrozenResNet18Extractor",
    "RESNET18_EMBEDDING_DIM",
    "RESNET18_VISUAL_CACHE_VERSION",
    "ResNet18FeatureCache",
    "ResNet18FeatureConfig",
    "ResNet18FeatureResult",
    "ResNet18Record",
    "build_argument_parser",
    "classification_metrics",
    "fit_train_only_pca",
    "main",
    "pool_reduced_sequence",
    "precompute_resnet18_features",
    "train_resnet18_pca_logistic_baseline",
    "transform_and_pool_sequences",
    "uniform_frame_indices",
]
