"""Measurable semantic cues for five-pillar short-video affect modelling.

This module deliberately separates *what is observed* from an affect
prediction.  It does not call a dark soundtrack ``negative`` or rapid motion
``violent``.  Instead it returns reproducible acoustic, lighting, spatial,
motion, and temporal measurements plus carefully worded descriptive cues.  A
model trained on LIRIS-ACCEDE can learn how those cues relate to its valence
labels.

The output is designed for a sequence fusion model.  For a clip with ``T``
sampled windows, every entry in ``features`` has shape ``(T, D)`` and dtype
``float32``; every entry in ``availability`` and ``reliability`` has shape
``(T,)`` and dtype ``float32``.  ``availability`` means that a modality is
present.  ``reliability`` is a continuous information-quality estimate.  For
example, a black or low-detail frame remains visually available but has low
colour/spatial reliability, while a mute track has audio availability zero.

Optional packages and models are imported only when they are used.  In
particular, importing this module never downloads a model.  The optional
ResNet-18 context encoder is disabled by default and, unless explicitly
allowed, only uses a checkpoint that is already present in the local torch
cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


# v3 fixes the optional-feature contract and is deliberately incompatible with
# v2 feature caches.  In particular, v2 recorded fixed dimensions even when
# PANNs or CLIP was enabled, which meant a cache could claim to contain a
# different representation from the arrays it actually held.
SCHEMA_VERSION = "semantic-five-pillar-v3"
# A v2 *baseline* checkpoint may still be used for upload inference because
# its five tensor widths and feature ordering are unchanged.  Caches are never
# read across this boundary; see the training cache's schema/signature check.
LEGACY_BASELINE_SCHEMA_VERSION = "semantic-five-pillar-v2"

# The source digest is included in the cache signature below.  The explicit
# version gives experiment manifests a readable identifier, while the digest
# makes a preprocessing implementation edit invalidate old arrays even when a
# developer forgets to bump this string.  It is intentionally extraction-only
# metadata: it never changes model weights or the public feature ordering.
EXTRACTOR_IMPLEMENTATION_VERSION = "2026-09-17.feature-contract-v1"

# These are context widths, not full pillar widths.  Spatial features always
# contain 16 engineered values; the baseline then appends an 8-D fixed ResNet
# projection, whereas CLIP appends its 512-D image embedding.
ENGINEERED_SPATIAL_DIM = 16
RESNET_CONTEXT_DIM = 8
CLIP_CONTEXT_DIM = 512
PANNS_CONTEXT_DIM = 527

# Keep this order stable: it is used by sequence-fusion models and cached data.
PILLAR_NAMES = ("audio", "color", "spatial", "motion", "temporal")
PILLAR_DIMS = {
    "audio": 20,
    "color": 16,
    "spatial": ENGINEERED_SPATIAL_DIM + RESNET_CONTEXT_DIM,
    "motion": 14,
    "temporal": 12,
}

# Human-readable feature layouts make cached arrays reviewable in a paper or
# experiment notebook.  They are descriptors, not labels or fixed rules.
PILLAR_FEATURES = {
    "audio": (
        "rms_mean", "rms_std", "dynamic_range", "zero_crossing_rate",
        "spectral_centroid", "spectral_centroid_std", "spectral_rolloff",
        "spectral_bandwidth", "spectral_flatness", "onset_strength",
        "onset_density", "global_tempo", "low_band_ratio", "mid_band_ratio",
        "high_band_ratio", "spectral_slope", "spectral_peakiness",
        "amplitude_modulation", "active_audio_fraction", "audio_present",
    ),
    "color": (
        "brightness", "brightness_std", "dark_pixel_fraction",
        "bright_pixel_fraction", "saturation", "saturation_std", "hue_sin",
        "hue_cos", "warm_cool_balance", "cool_pixel_fraction",
        "colorfulness", "luminance_contrast", "luminance_entropy",
        "warm_pixel_fraction", "daylight_proxy", "nighttime_proxy",
    ),
    "spatial": (
        "edge_density", "edge_orientation_entropy", "texture_detail",
        "luminance_entropy", "center_saliency_proxy", "horizontal_balance",
        "vertical_balance", "rule_of_thirds_proxy", "contour_density",
        "largest_region_fraction", "face_count_proxy", "face_area_proxy",
        "aspect_ratio", "vertical_lighting_gradient", "center_color_contrast",
        "pretrained_context_available", "resnet_projection_0",
        "resnet_projection_1", "resnet_projection_2", "resnet_projection_3",
        "resnet_projection_4", "resnet_projection_5", "resnet_projection_6",
        "resnet_projection_7",
    ),
    "motion": (
        "mean_flow_speed", "p90_flow_speed", "flow_speed_std",
        "active_motion_fraction", "global_flow_coherence",
        "direction_entropy", "global_translation_proxy", "motion_acceleration",
        "mean_horizontal_flow", "mean_vertical_flow", "localized_motion_fraction",
        "flow_spread", "kinematic_intensity_proxy", "motion_pair_valid",
    ),
    "temporal": (
        "relative_time", "luminance_change", "color_change",
        "brightness_delta", "saturation_delta", "cut_like_change_proxy",
        "motion_speed", "motion_acceleration", "visual_activity",
        "cumulative_visual_change", "audio_energy_change", "audio_visual_sync_proxy",
    ),
}


@dataclass(frozen=True)
class SemanticPillarConfig:
    """Configuration shared by training-time and upload-time extraction.

    ``sampled_windows`` is an upper bound.  Very short clips return fewer
    windows rather than inventing duplicate observations; ordinary LIRIS clips
    return the configured number.  Sequence models should pad batches and use
    their sequence mask when mixing arbitrary upload durations.
    """

    max_duration_seconds: float = 15.0
    sampled_windows: int = 16
    analysis_size: int = 192
    flow_size: int = 128
    audio_sample_rate: int = 16_000
    silence_rms_threshold: float = 1e-4
    use_pretrained_spatial: bool = False
    allow_model_download: bool = False
    spatial_device: str = "cpu"
    use_panns: bool = False
    use_clip: bool = False
    # Optional pretrained branches must be explicitly local.  This prevents a
    # cache job or upload inference from unexpectedly downloading hundreds of
    # megabytes of weights and, more importantly, makes the representation in
    # an experiment reproducible.
    panns_checkpoint_path: str | None = None
    clip_model_id: str = "openai/clip-vit-base-patch32"
    clip_revision: str | None = None



class SemanticPillarExtractor:
    """Extract five synchronized, descriptive sequences from a short video.

    The standard result has the following schema::

        {
          "schema_version": "semantic-five-pillar-v3",
          "video": {...},
          "features": {pillar: float32 array shaped (T, extractor.pillar_dims[pillar])},
          "availability": {pillar: float32 array shaped (T,)},
          "reliability": {pillar: float32 array shaped (T,)},
          "evidence": {pillar: JSON-safe aggregate measurements},
          "semantic_labels": {pillar: list[list[str]] of length T},
          "temporal_inputs": {"timestamps_seconds": float32 array shaped (T,)},
        }

    ``semantic_labels`` are descriptive report cues such as
    ``"low_light"`` or ``"energetic_acoustic_pattern"``.  They are never
    sentiment decisions and the motion cues intentionally do not claim that an
    action is harmful or violent.
    """

    def __init__(self, config: SemanticPillarConfig | None = None):
        self.config = config or SemanticPillarConfig()
        if self.config.sampled_windows < 1:
            raise ValueError("sampled_windows must be at least 1")
        if self.config.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")

        # All optional resources are lazy.  A failed optional resource is
        # remembered so a 9,800-video cache run does not retry it per clip.
        self._face_detector: Any | None = None
        self._face_detector_checked = False
        self._spatial_model: Any | None = None
        self._spatial_torch: Any | None = None
        self._spatial_categories: list[str] = []
        self._spatial_model_checked = False
        self._spatial_model_status = "disabled"
        
        self._audio_model: Any | None = None
        self._audio_model_checked = False
        self._audio_model_status = "disabled"
        
        rng = np.random.default_rng(20_260_915)
        self._resnet_projection = (
            rng.standard_normal((512, 8)).astype(np.float32) / np.sqrt(512.0)
        )

    @property
    def pillar_dims(self) -> dict[str, int]:
        """Return the exact per-pillar widths produced by this configuration.

        ``PILLAR_DIMS`` is the v2/v3 baseline contract.  Optional pretrained
        branches replace the baseline context tail rather than being appended
        to the already-expanded pillar.  Keeping this computation in one
        place prevents the extractor, cache validator, trainer, and upload
        inference from disagreeing about their tensor shapes.
        """

        dims = dict(PILLAR_DIMS)
        if self.config.use_panns:
            dims["audio"] = PILLAR_DIMS["audio"] + PANNS_CONTEXT_DIM
        if self.config.use_clip:
            dims["spatial"] = CLIP_CONTEXT_DIM
        return dims

    @property
    def pillar_features(self) -> dict[str, tuple[str, ...]]:
        """Return the feature layout matching :attr:`pillar_dims` exactly."""

        layouts = {name: tuple(PILLAR_FEATURES[name]) for name in PILLAR_NAMES}
        if self.config.use_panns:
            layouts["audio"] = (
                *layouts["audio"],
                *(f"panns_audio_tag_{index:03d}" for index in range(PANNS_CONTEXT_DIM)),
            )
        if self.config.use_clip:
            layouts["spatial"] = tuple(f"clip_image_embedding_{index:03d}" for index in range(CLIP_CONTEXT_DIM))
        dims = self.pillar_dims
        for name in PILLAR_NAMES:
            if len(layouts[name]) != dims[name]:  # Defensive developer invariant.
                raise RuntimeError(
                    f"Feature layout for {name!r} has {len(layouts[name])} values, "
                    f"but the configured width is {dims[name]}."
                )
        return layouts


    def extract(self, video_path: str | Path, *, truncate_to_limit: bool = False) -> dict[str, Any]:
        """Return synchronized semantic-pillar sequences for ``video_path``.

        Raises:
            FileNotFoundError: the path does not exist.
            ValueError: the video cannot be decoded, has no frames, or is over
            the configured short-video duration limit.  ``truncate_to_limit``
            is reserved for an internal training-data repair path: it analyzes
            only the first configured-duration seconds of a known overlong
            source clip.  Upload inference always leaves it disabled and
            therefore rejects an overlong upload.

        A missing audio decoder/track is not an error.  It produces zero audio
        features, zero audio availability, and leaves the other four pillars
        usable.
        """

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file was not found: {path}")

        frames, motion_frames, timestamps, motion_deltas, metadata = self._read_video_windows(
            path,
            truncate_to_limit=truncate_to_limit,
        )
        color_features, color_reliability, color_evidence, color_labels = self._color_pillar(frames)
        spatial_features, spatial_reliability, spatial_evidence, spatial_labels = self._spatial_pillar(
            frames,
            metadata["aspect_ratio"],
        )
        motion_features, motion_available, motion_reliability, motion_evidence, motion_labels = self._motion_pillar(
            frames,
            motion_frames,
            motion_deltas,
            spatial_reliability,
        )
        audio_features, audio_available, audio_reliability, audio_evidence, audio_labels = self._audio_pillar(
            path,
            timestamps,
            float(metadata["duration_seconds"]),
        )
        temporal_features, temporal_available, temporal_reliability, temporal_evidence, temporal_labels = self._temporal_pillar(
            timestamps,
            color_features,
            color_reliability,
            spatial_reliability,
            motion_features,
            motion_available,
            motion_reliability,
            audio_features,
            audio_available,
            audio_reliability,
        )

        features = {
            "audio": audio_features,
            "color": color_features,
            "spatial": spatial_features,
            "motion": motion_features,
            "temporal": temporal_features,
        }
        availability = {
            "audio": audio_available,
            "color": np.ones(len(frames), dtype=np.float32),
            "spatial": np.ones(len(frames), dtype=np.float32),
            "motion": motion_available,
            "temporal": temporal_available,
        }
        reliability = {
            "audio": audio_reliability,
            "color": color_reliability,
            "spatial": spatial_reliability,
            "motion": motion_reliability,
            "temporal": temporal_reliability,
        }
        self._validate_output(features, availability, reliability)

        metadata["sampled_windows"] = int(len(frames))
        evidence = {
            "audio": audio_evidence,
            "color": color_evidence,
            "spatial": spatial_evidence,
            "motion": motion_evidence,
            "temporal": temporal_evidence,
            "notes": {
                "semantics": (
                    "Labels are descriptive observations; a trained fusion model, "
                    "not this extractor, predicts induced affect."
                ),
                "motion": (
                    "Kinematic intensity is a speed/change proxy. It is not a "
                    "violence, safety, or harm detector."
                ),
                "reliability": (
                    "Low reliability denotes limited usable evidence, not a "
                    "negative affect label."
                ),
            },
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "extractor": {
                "name": type(self).__name__,
                "signature": self.cache_signature(),
                "config": self.config_metadata(),
            },
            "video": metadata,
            "features": features,
            "availability": availability,
            "reliability": reliability,
            "evidence": evidence,
            "semantic_labels": {
                "audio": audio_labels,
                "color": color_labels,
                "spatial": spatial_labels,
                "motion": motion_labels,
                "temporal": temporal_labels,
            },
            "temporal_inputs": {
                "timestamps_seconds": timestamps.astype(np.float32),
                "motion_pair_seconds": motion_deltas.astype(np.float32),
                "sequence_length": int(len(frames)),
            },
        }

    def config_metadata(self) -> dict[str, Any]:
        """Return JSON-safe settings that must match for cached features.

        Store this together with cached feature arrays.  A change to any value
        changes :meth:`cache_signature`, making stale feature caches easy to
        identify rather than silently mixing incompatible preprocessing.
        """

        config = asdict(self.config)
        layouts = self.pillar_features
        return {
            "schema_version": SCHEMA_VERSION,
            "implementation": {
                "version": EXTRACTOR_IMPLEMENTATION_VERSION,
                "source_sha256": self._implementation_fingerprint(),
            },
            "config": config,
            "pillar_dims": self.pillar_dims,
            "pillar_features": {name: list(layouts[name]) for name in PILLAR_NAMES},
            "local_resource_fingerprints": {
                "panns_checkpoint": self._local_file_fingerprint(
                    self.config.panns_checkpoint_path
                ),
            },
        }

    def cache_signature(self) -> str:
        """Stable signature for cache invalidation across extraction settings."""

        encoded = json.dumps(self.config_metadata(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    @staticmethod
    def _implementation_fingerprint() -> str:
        """Return a source digest for conservative cache invalidation.

        Wheels normally ship the source file, but use the readable version as
        a stable fallback for source-stripped deployments.  No network or
        model resource is consulted here.
        """

        try:
            payload = Path(__file__).read_bytes()
        except OSError:
            payload = EXTRACTOR_IMPLEMENTATION_VERSION.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    @staticmethod
    def _local_file_fingerprint(path_value: str | None) -> dict[str, Any]:
        """Describe an explicit local optional-model file without loading it."""

        if not path_value:
            return {"status": "not_configured"}
        path = Path(path_value).expanduser()
        try:
            stat = path.stat()
        except OSError:
            return {"status": "missing", "path": str(path)}
        return {
            "status": "present",
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    # ------------------------------------------------------------------
    # Video decoding / synchronized sample windows
    # ------------------------------------------------------------------
    def _read_video_windows(
        self,
        path: Path,
        *,
        truncate_to_limit: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
        cv2 = self._cv2()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Could not decode video: {path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                fps = 25.0
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_count > 0:
                source_duration = frame_count / fps
                was_truncated = source_duration > self.config.max_duration_seconds + 1e-6
                if was_truncated and not truncate_to_limit:
                    raise ValueError(
                        f"Video is {source_duration:.2f}s; this extractor accepts clips up to "
                        f"{self.config.max_duration_seconds:.0f}s."
                    )
                if was_truncated:
                    # Keep only complete source frames inside the configured
                    # short-video limit.  This is used solely for two known
                    # LIRIS duration anomalies; normal upload inference still
                    # rejects videos longer than the public 15-second limit.
                    frame_count = max(1, min(frame_count, int(np.floor(self.config.max_duration_seconds * fps))))
                duration = frame_count / fps
                raw_frames, sample_indices, partner_indices = self._seek_sample_windows(
                    capture,
                    frame_count,
                    fps,
                )
                frames = [self._resize_frame(frame) for frame in raw_frames[0]]
                motion_frames = [self._resize_frame(frame) for frame in raw_frames[1]]
                timestamps = sample_indices.astype(np.float32) / float(fps)
                motion_deltas = np.abs(partner_indices - sample_indices).astype(np.float32) / float(fps)
            else:
                # Some containers do not expose frame metadata.  Decode once,
                # still enforcing the short-video limit before retaining more
                # than a small bounded amount of data.
                decoded: list[np.ndarray] = []
                max_frames = int(np.ceil(self.config.max_duration_seconds * fps)) + 2
                while len(decoded) <= max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    decoded.append(frame)
                if not decoded:
                    raise ValueError(f"No readable frames found in: {path}")
                was_truncated = len(decoded) > max_frames
                if was_truncated and not truncate_to_limit:
                    raise ValueError(
                        f"Video exceeds the {self.config.max_duration_seconds:.0f}s duration limit."
                    )
                if was_truncated:
                    decoded = decoded[:max_frames]
                frame_count = len(decoded)
                duration = frame_count / fps
                source_duration = duration if not was_truncated else float("nan")
                if width <= 0 or height <= 0:
                    height, width = decoded[0].shape[:2]
                raw_frames, sample_indices, partner_indices = self._sample_decoded_windows(decoded, fps)
                frames = [self._resize_frame(frame) for frame in raw_frames[0]]
                motion_frames = [self._resize_frame(frame) for frame in raw_frames[1]]
                timestamps = sample_indices.astype(np.float32) / float(fps)
                motion_deltas = np.abs(partner_indices - sample_indices).astype(np.float32) / float(fps)

            if not frames:
                raise ValueError(f"No readable frames found in: {path}")
            if width <= 0 or height <= 0:
                height, width = frames[0].shape[:2]
            metadata = {
                "path": str(path),
                "duration_seconds": self._float(duration),
                "source_duration_seconds": self._float(source_duration),
                "truncated_to_limit": bool(was_truncated),
                "fps": self._float(fps),
                "frame_count": int(frame_count),
                "frame_width": int(width),
                "frame_height": int(height),
                "aspect_ratio": self._float(float(width) / max(float(height), 1.0)),
            }
            return frames, motion_frames, timestamps, motion_deltas, metadata
        finally:
            capture.release()

    def _seek_sample_windows(
        self,
        capture: Any,
        frame_count: int,
        fps: float,
    ) -> tuple[tuple[list[np.ndarray], list[np.ndarray]], np.ndarray, np.ndarray]:
        """Seek representative + near-future frames for every analysis window."""

        cv2 = self._cv2()
        sample_count = min(self.config.sampled_windows, max(frame_count, 1))
        # A local pair is much more meaningful for flow than comparing frames
        # one second apart.  It is constrained for very short clips.
        max_offset_seconds = min(0.25, max(1.0 / fps, frame_count / fps / (2 * sample_count)))
        offset = max(1, int(round(max_offset_seconds * fps))) if frame_count > 1 else 0
        indices = np.linspace(0, frame_count - 1, sample_count, dtype=np.int32)
        partners = np.empty_like(indices)
        source_frames: list[np.ndarray] = []
        partner_frames: list[np.ndarray] = []
        for index in indices.tolist():
            if frame_count <= 1:
                partner = index
            elif index + offset < frame_count:
                partner = index + offset
            else:
                partner = max(0, index - offset)
            source = self._read_frame_at(capture, int(index), cv2)
            paired = self._read_frame_at(capture, int(partner), cv2)
            if source is None or paired is None:
                raise ValueError("Unable to seek one or more sampled video frames")
            source_frames.append(source)
            partner_frames.append(paired)
            partners[len(source_frames) - 1] = partner
        return (source_frames, partner_frames), indices, partners

    def _sample_decoded_windows(
        self,
        decoded: list[np.ndarray],
        fps: float,
    ) -> tuple[tuple[list[np.ndarray], list[np.ndarray]], np.ndarray, np.ndarray]:
        frame_count = len(decoded)
        sample_count = min(self.config.sampled_windows, max(frame_count, 1))
        max_offset_seconds = min(0.25, max(1.0 / fps, frame_count / fps / (2 * sample_count)))
        offset = max(1, int(round(max_offset_seconds * fps))) if frame_count > 1 else 0
        indices = np.linspace(0, frame_count - 1, sample_count, dtype=np.int32)
        partners = np.asarray(
            [
                index if frame_count <= 1 else (index + offset if index + offset < frame_count else max(0, index - offset))
                for index in indices.tolist()
            ],
            dtype=np.int32,
        )
        return (
            ([decoded[int(index)] for index in indices], [decoded[int(index)] for index in partners]),
            indices,
            partners,
        )

    @staticmethod
    def _read_frame_at(capture: Any, frame_index: int, cv2: Any) -> np.ndarray | None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        return frame if ok and frame is not None else None

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        cv2 = self._cv2()
        return cv2.resize(
            frame,
            (self.config.analysis_size, self.config.analysis_size),
            interpolation=cv2.INTER_AREA,
        )

    # ------------------------------------------------------------------
    # Pillar 1: audio semantics / acoustics.  No transcript or Whisper.
    # ------------------------------------------------------------------
    def _audio_pillar(
        self,
        path: Path,
        timestamps: np.ndarray,
        duration: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[list[str]]]:
        t = len(timestamps)
        # The unavailable path must preserve the configured representation
        # width.  Returning the 20-D baseline here under --use-panns used to
        # poison caches and fail only later in batching/inference.
        empty = np.zeros((t, self.pillar_dims["audio"]), dtype=np.float32)
        unavailable = np.zeros(t, dtype=np.float32)
        labels = [["audio_unavailable"] for _ in range(t)]
        default_evidence = {
            "audio_status": "unavailable",
            "availability_fraction": 0.0,
            "mean_rms": 0.0,
            "mean_tempo_bpm": 0.0,
            "mean_spectral_centroid_hz": 0.0,
            "note": "No decodable non-silent audio was available.",
        }
        if self.config.use_panns:
            default_evidence["panns_status"] = self._audio_model_status
        librosa = self._librosa()
        if librosa is None:
            default_evidence["audio_status"] = "dependency_unavailable"
            default_evidence["note"] = "librosa/audio decoding is unavailable in this runtime."
            return empty, unavailable, unavailable.copy(), default_evidence, labels

        waveform, sample_rate, decoder_status, decoder_error = self._decode_audio(
            librosa,
            path,
        )
        if waveform is None or sample_rate is None:
            # Do not describe a decoder failure as a mute video.  The bundled
            # FFmpeg route distinguishes a container with no audio stream from
            # a real decode error; the optional MoviePy probe is retained as a
            # last best-effort fallback for runtimes without it.
            stream_status = decoder_status or self._probe_audio_stream(path)
            default_evidence["audio_status"] = stream_status or "audio_decode_failure"
            if decoder_error:
                default_evidence["audio_error_type"] = decoder_error
            default_evidence["note"] = "Audio could not be decoded; this is not treated as a silent track."
            return empty, unavailable, unavailable.copy(), default_evidence, labels

        if waveform.size < max(16, int(sample_rate * 0.03)):
            default_evidence["audio_status"] = "empty_audio_stream"
            default_evidence["note"] = "An audio stream decoded to fewer than 30 milliseconds of samples."
            return empty, unavailable, unavailable.copy(), default_evidence, labels
        global_rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        if not np.isfinite(global_rms) or global_rms < self.config.silence_rms_threshold:
            default_evidence["audio_status"] = "silent_audio_track"
            default_evidence["note"] = "The decoded soundtrack is silent or below the audio threshold."
            return empty, unavailable, unavailable.copy(), default_evidence, labels

        tempo = self._global_tempo(librosa, waveform, int(sample_rate))
        boundaries = self._window_boundaries(timestamps, duration)
        feature_rows: list[np.ndarray] = []
        availability_rows: list[float] = []
        reliability_rows: list[float] = []
        labels = []
        rms_values: list[float] = []
        centroid_values: list[float] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            start_sample = max(0, int(round(start * sample_rate)))
            end_sample = min(len(waveform), max(start_sample, int(round(end * sample_rate))))
            segment = waveform[start_sample:end_sample]
            row, available, reliability, row_labels, row_evidence = self._audio_window(
                librosa,
                segment,
                int(sample_rate),
                tempo,
            )
            feature_rows.append(row)
            availability_rows.append(float(available))
            reliability_rows.append(float(reliability))
            labels.append(row_labels)
            rms_values.append(float(row_evidence["rms"]))
            centroid_values.append(float(row_evidence["centroid_hz"]))

        features = np.asarray(feature_rows, dtype=np.float32)
        availability = np.asarray(availability_rows, dtype=np.float32)
        reliability = np.asarray(reliability_rows, dtype=np.float32)
        evidence = {
            "audio_status": "audio_present" if np.any(availability > 0.5) else "audio_feature_extraction_failed",
            "availability_fraction": self._float(float(availability.mean())),
            "mean_rms": self._float(float(np.mean(rms_values))),
            "mean_tempo_bpm": self._float(tempo),
            "mean_spectral_centroid_hz": self._float(float(np.mean(centroid_values))),
            "note": (
                "Audio cues describe acoustic energy, timbre, and rhythm; they do not "
                "assign happy/sad meaning to a sound."
            ),
        }
        if self.config.use_panns:
            evidence["panns_status"] = self._audio_model_status
        return features, availability, reliability, evidence, labels

    def _decode_audio(
        self,
        librosa: Any,
        path: Path,
    ) -> tuple[np.ndarray | None, int | None, str | None, str | None]:
        """Decode mono PCM without relying on a system-wide FFmpeg install.

        MP4 files are not normally readable by libsndfile, and ``librosa``'s
        deprecated audioread fallback only discovers FFmpeg when it happens to
        be on ``PATH``.  ``imageio-ffmpeg`` ships a local binary, so prefer it
        for video containers and retain librosa as a fallback for environments
        that do not install that optional helper.
        """

        waveform, status, error = self._decode_audio_with_bundled_ffmpeg(
            path,
            sample_rate=self.config.audio_sample_rate,
            max_duration_seconds=self.config.max_duration_seconds,
        )
        if waveform is not None:
            return waveform, self.config.audio_sample_rate, None, None
        if status in {"no_audio_stream", "empty_audio_stream"}:
            return None, None, status, error

        try:
            fallback, fallback_rate = librosa.load(
                str(path),
                sr=self.config.audio_sample_rate,
                mono=True,
                duration=self.config.max_duration_seconds,
            )
            return np.asarray(fallback, dtype=np.float32), int(fallback_rate), None, None
        except Exception as fallback_error:
            # Keep the FFmpeg failure useful for debugging while avoiding an
            # unbounded decoder stderr dump in cached metadata.
            detail = error or type(fallback_error).__name__
            return None, None, status or self._probe_audio_stream(path), detail

    @staticmethod
    def _decode_audio_with_bundled_ffmpeg(
        path: Path,
        *,
        sample_rate: int,
        max_duration_seconds: float,
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        """Return PCM through imageio-ffmpeg's private local executable.

        The helper is deliberately optional: missing package/binary falls back
        to librosa and ultimately produces an explicit decoder status rather
        than falsely masking the audio pillar as mute.
        """

        try:
            import imageio_ffmpeg

            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None, None, None

        command = [
            str(executable),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-t",
            f"{float(max_duration_seconds):.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(int(sample_rate)),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=max(30.0, float(max_duration_seconds) + 15.0),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return None, "audio_decode_failure", type(error).__name__

        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if completed.returncode != 0:
            lowered = stderr.lower()
            if "matches no streams" in lowered or "stream specifier" in lowered:
                return None, "no_audio_stream", None
            return None, "audio_decode_failure", (
                stderr[-240:] if stderr else f"ffmpeg_exit_{completed.returncode}"
            )

        # An empty successful PCM response represents an empty track.  It is
        # intentionally distinct from a decoded silent waveform below.
        if len(completed.stdout) < np.dtype("<f4").itemsize:
            return None, "empty_audio_stream", None
        waveform = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32, copy=True)
        if waveform.size == 0 or not np.all(np.isfinite(waveform)):
            return None, "audio_decode_failure", "invalid_pcm_output"
        return waveform, None, None

    @staticmethod
    def _probe_audio_stream(path: Path) -> str | None:
        """Best-effort no-stream probe used only after an audio decode failure.

        MoviePy is optional and imported lazily.  ``None`` means that the
        probe is unavailable or inconclusive, not that the video is silent.
        """

        try:
            from moviepy.editor import VideoFileClip
        except Exception:
            return None
        clip: Any | None = None
        try:
            clip = VideoFileClip(str(path), audio=True)
            return "audio_decode_failure" if clip.audio is not None else "no_audio_stream"
        except Exception:
            return None
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass

    def _audio_window(
        self,
        librosa: Any,
        segment: np.ndarray,
        sample_rate: int,
        tempo: float,
    ) -> tuple[np.ndarray, bool, float, list[str], dict[str, float]]:
        audio_dim = self.pillar_dims["audio"]
        zeros = np.zeros(audio_dim, dtype=np.float32)
        if segment.size < max(64, int(sample_rate * 0.03)):
            return zeros, False, 0.0, ["audio_unavailable"], {"rms": 0.0, "centroid_hz": 0.0}
        segment = np.nan_to_num(np.asarray(segment, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        rms_signal = float(np.sqrt(np.mean(np.square(segment, dtype=np.float64))))
        if rms_signal < self.config.silence_rms_threshold:
            return zeros, False, 0.0, ["silent_window"], {"rms": rms_signal, "centroid_hz": 0.0}

        # A short window can be padded for a stable FFT, without changing its
        # RMS/missing-audio decision.
        n_fft = 512
        if segment.size < n_fft:
            segment = np.pad(segment, (0, n_fft - segment.size))
        hop = 256
        try:
            rms = librosa.feature.rms(y=segment, frame_length=n_fft, hop_length=hop)[0]
            zcr = librosa.feature.zero_crossing_rate(segment, frame_length=n_fft, hop_length=hop)[0]
            centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate, n_fft=n_fft, hop_length=hop)[0]
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sample_rate, n_fft=n_fft, hop_length=hop)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sample_rate, n_fft=n_fft, hop_length=hop)[0]
            flatness = librosa.feature.spectral_flatness(y=segment, n_fft=n_fft, hop_length=hop)[0]
            onset = librosa.onset.onset_strength(y=segment, sr=sample_rate, hop_length=hop)
            mel = librosa.feature.melspectrogram(y=segment, sr=sample_rate, n_fft=n_fft, hop_length=hop, n_mels=24, power=2.0)
        except Exception:
            return zeros, False, 0.0, ["audio_feature_error"], {"rms": rms_signal, "centroid_hz": 0.0}

        mel_energy = np.mean(mel, axis=1) + 1e-8
        low = float(np.sum(mel_energy[:8]) / np.sum(mel_energy))
        mid = float(np.sum(mel_energy[8:17]) / np.sum(mel_energy))
        high = float(np.sum(mel_energy[17:]) / np.sum(mel_energy))
        amplitude_modulation = self._clip01(float(np.std(rms) / (np.mean(rms) + 1e-8)))
        onset_mean = self._clip01(float(np.tanh(np.mean(onset) / 3.0))) if onset.size else 0.0
        onset_density = float(np.mean(onset > (np.mean(onset) + 0.5 * np.std(onset)))) if onset.size else 0.0
        active_fraction = float(np.mean(rms > max(self.config.silence_rms_threshold, np.mean(rms) * 0.25)))
        peakiness = self._clip01(float(np.max(mel_energy) / np.sum(mel_energy) * len(mel_energy)))
        spectrum_half = max(sample_rate / 2.0, 1.0)
        base_features = [
            self._clip01(float(np.mean(rms)) * 12.0),
            self._clip01(float(np.std(rms)) * 12.0),
            self._clip01(float(np.percentile(rms, 95) - np.percentile(rms, 5)) * 12.0),
            self._clip01(float(np.mean(zcr))),
            self._clip01(float(np.mean(centroid)) / spectrum_half),
            self._clip01(float(np.std(centroid)) / spectrum_half),
            self._clip01(float(np.mean(rolloff)) / spectrum_half),
            self._clip01(float(np.mean(bandwidth)) / spectrum_half),
            self._clip01(float(np.mean(flatness))),
            onset_mean,
            self._clip01(onset_density),
            self._clip01(float(tempo) / 300.0),
            self._clip01(low),
            self._clip01(mid),
            self._clip01(high),
            float(np.clip(high - low, -1.0, 1.0)),
            peakiness,
            amplitude_modulation,
            self._clip01(active_fraction),
            1.0,
        ]
        
        model: Any | None = None
        panns_used = False
        if self.config.use_panns:
            model = self._maybe_load_audio_model()
            panns_scores = (
                self._panns_clipwise_scores(model, segment, sample_rate)
                if model is not None
                else None
            )
            if panns_scores is None:
                # Keep the row shape deterministic even when the requested
                # local optional checkpoint is unavailable.  Metadata records
                # that this context tail is zero-filled; callers must not
                # mistake it for a successful PANNs extraction.
                panns_scores = np.zeros(PANNS_CONTEXT_DIM, dtype=np.float32)
            else:
                panns_used = True
            base_features.extend(panns_scores.tolist())

        row = np.asarray(base_features, dtype=np.float32)
        if row.shape != (audio_dim,):
            raise RuntimeError(
                f"Audio feature contract produced {row.shape}; expected ({audio_dim},)."
            )

        labels: list[str] = ["audio_present"]
        if row[0] < 0.08:
            labels.append("quiet_soundtrack")
        elif row[0] > 0.35:
            labels.append("loud_soundtrack")
        if row[4] < 0.24 or row[15] < -0.18:
            labels.append("dark_timbre_proxy")
        elif row[4] > 0.48 or row[15] > 0.18:
            labels.append("bright_timbre_proxy")
        if row[9] > 0.24 or (tempo >= 105.0 and row[0] > 0.10):
            labels.append("energetic_acoustic_pattern")
        elif row[0] < 0.12 and row[9] < 0.12:
            labels.append("calm_acoustic_pattern")
        if row[8] > 0.35:
            labels.append("noisy_or_percussive_texture_proxy")
        if self.config.use_panns and panns_used:
            labels.append("panns_audio_tags")
            self._audio_model_status = "cached_pretrained_panns"
        elif self.config.use_panns:
            labels.append("panns_audio_tags_unavailable")

        reliability = self._clip01(0.5 * min(1.0, rms_signal / 0.02) + 0.3 * active_fraction + 0.2 * min(1.0, segment.size / sample_rate))
        return row, True, reliability, labels, {"rms": rms_signal, "centroid_hz": float(np.mean(centroid))}

    def _maybe_load_audio_model(self) -> Any | None:
        if self._audio_model_checked:
            return self._audio_model
        self._audio_model_checked = True
        if not self.config.use_panns:
            self._audio_model_status = "disabled"
            return None
        checkpoint_value = self.config.panns_checkpoint_path
        if not checkpoint_value:
            self._audio_model_status = "panns_checkpoint_not_configured"
            return None
        checkpoint_path = Path(checkpoint_value).expanduser()
        if not checkpoint_path.is_file():
            self._audio_model_status = "panns_checkpoint_missing"
            return None
        try:
            from panns_inference import AudioTagging
            # Passing an existing explicit path is deliberately the only load
            # route.  ``checkpoint_path=None`` can make panns-inference fetch
            # weights from the network, which is unsafe for reproducible cache
            # generation and prohibited for upload inference.
            model = AudioTagging(
                checkpoint_path=str(checkpoint_path),
                device=self.config.spatial_device,
            )
            self._audio_model = model
            self._audio_model_status = "cached_pretrained_panns"
        except Exception as error:
            self._audio_model = None
            self._audio_model_status = f"pretrained_audio_unavailable:{type(error).__name__}"
        return self._audio_model

    def _panns_clipwise_scores(
        self,
        model: Any,
        segment: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray | None:
        """Return exactly 527 clipwise PANNs scores, independent of API order.

        panns-inference releases have returned their clipwise scores and
        embedding in different tuple positions.  Select by width instead of
        assuming one position; an unexpected response is reported as an
        unavailable optional branch rather than changing the feature width.
        """

        try:
            import librosa as resample_librosa

            if sample_rate != 32_000:
                sampled = resample_librosa.resample(
                    segment,
                    orig_sr=sample_rate,
                    target_sr=32_000,
                )
            else:
                sampled = segment
            output = model.inference(
                np.asarray(sampled, dtype=np.float32).reshape(1, -1)
            )
            candidates: list[Any]
            if isinstance(output, dict):
                candidates = [
                    output.get("clipwise_output"),
                    output.get("clipwise_scores"),
                    output.get("output"),
                ]
            elif isinstance(output, (tuple, list)):
                candidates = list(output)
            else:
                candidates = [output]
            for candidate in candidates:
                if candidate is None:
                    continue
                if hasattr(candidate, "detach"):
                    candidate = candidate.detach().cpu().numpy()
                values = np.asarray(candidate, dtype=np.float32)
                if values.ndim >= 1 and values.shape[-1] == PANNS_CONTEXT_DIM:
                    values = values.reshape(-1, PANNS_CONTEXT_DIM)
                    if values.shape[0] >= 1:
                        result = values[0]
                        if np.all(np.isfinite(result)):
                            return result.astype(np.float32, copy=False)
            self._audio_model_status = "panns_inference_failed:unexpected_output_shape"
        except Exception as error:
            self._audio_model_status = f"panns_inference_failed:{type(error).__name__}"
        return None

    @staticmethod
    def _global_tempo(librosa: Any, waveform: np.ndarray, sample_rate: int) -> float:
        try:
            tempo, _ = librosa.beat.beat_track(y=waveform, sr=sample_rate, hop_length=512)
            values = np.asarray(tempo).reshape(-1)
            return float(values[0]) if values.size and np.isfinite(values[0]) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _window_boundaries(timestamps: np.ndarray, duration: float) -> np.ndarray:
        if len(timestamps) == 1:
            return np.asarray([0.0, max(duration, 0.0)], dtype=np.float32)
        boundaries = np.empty(len(timestamps) + 1, dtype=np.float32)
        boundaries[0] = 0.0
        boundaries[-1] = max(float(duration), float(timestamps[-1]))
        boundaries[1:-1] = (timestamps[:-1] + timestamps[1:]) / 2.0
        # Decoder timestamps can occasionally be repeated; keep every segment
        # non-negative and let a zero-length segment become unavailable.
        return np.maximum.accumulate(boundaries)

    # ------------------------------------------------------------------
    # Pillar 2: colour / lighting.
    # ------------------------------------------------------------------
    def _color_pillar(
        self,
        frames: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[list[str]]]:
        cv2 = self._cv2()
        rows: list[np.ndarray] = []
        reliabilities: list[float] = []
        labels_all: list[list[str]] = []
        brightnesses: list[float] = []
        warm_cool: list[float] = []
        day_proxies: list[float] = []
        night_proxies: list[float] = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            hue = hsv[..., 0] / 180.0 * (2.0 * np.pi)
            saturation = hsv[..., 1] / 255.0
            value = hsv[..., 2] / 255.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            warmth = float(np.mean(red - blue))
            colorfulness = self._colorfulness(frame)
            contrast = float(np.std(gray))
            entropy = self._histogram_entropy(gray, 32)
            dark_fraction = float(np.mean(value < 0.25))
            bright_fraction = float(np.mean(value > 0.75))
            cool_fraction = float(np.mean(blue > red + 0.06))
            warm_fraction = float(np.mean(red > blue + 0.06))
            # These are explicitly illumination proxies, not claims about the
            # actual time or location of the filmed scene.
            daylight = self._clip01(0.55 * float(np.mean(value)) + 0.25 * (1.0 - dark_fraction) + 0.20 * contrast)
            nighttime = self._clip01(0.70 * dark_fraction + 0.30 * (1.0 - float(np.mean(value))))
            row = np.asarray(
                [
                    float(np.mean(value)), float(np.std(value)), dark_fraction,
                    bright_fraction, float(np.mean(saturation)), float(np.std(saturation)),
                    float(np.mean(np.sin(hue))), float(np.mean(np.cos(hue))), warmth,
                    cool_fraction, self._clip01(colorfulness / 95.0), contrast, entropy,
                    warm_fraction, daylight, nighttime,
                ],
                dtype=np.float32,
            )
            # A flat black/white frame has low colour information.  It remains
            # available, and no sentiment label is inferred from that fact.
            exposure = self._clip01(min(float(np.mean(value)), 1.0 - float(np.mean(value))) / 0.25)
            reliability = self._clip01(
                0.45 * min(1.0, contrast / 0.12)
                + 0.30 * min(1.0, float(np.mean(saturation)) / 0.25)
                + 0.25 * exposure
            )
            labels: list[str] = []
            if nighttime > 0.60:
                labels.append("nighttime_like_lighting_proxy")
            elif daylight > 0.62:
                labels.append("daylight_like_lighting_proxy")
            else:
                labels.append("mid_light")
            if warmth > 0.045:
                labels.append("warm_palette")
            elif warmth < -0.045:
                labels.append("cool_palette")
            if float(np.mean(saturation)) < 0.20:
                labels.append("muted_palette")
            elif float(np.mean(saturation)) > 0.52:
                labels.append("vivid_palette")
            if contrast > 0.18:
                labels.append("high_luminance_contrast")
            elif contrast < 0.045:
                labels.append("low_visual_detail")
            rows.append(row)
            reliabilities.append(reliability)
            labels_all.append(labels)
            brightnesses.append(float(row[0]))
            warm_cool.append(warmth)
            day_proxies.append(daylight)
            night_proxies.append(nighttime)

        evidence = {
            "mean_brightness": self._float(float(np.mean(brightnesses))),
            "mean_warm_cool_balance": self._float(float(np.mean(warm_cool))),
            "mean_daylight_proxy": self._float(float(np.mean(day_proxies))),
            "mean_nighttime_proxy": self._float(float(np.mean(night_proxies))),
            "note": "Day/night and warm/cool entries are lighting/palette proxies, not scene facts.",
        }
        return np.asarray(rows, dtype=np.float32), np.asarray(reliabilities, dtype=np.float32), evidence, labels_all

    # ------------------------------------------------------------------
    # Pillar 3: spatial scene / composition context.
    # ------------------------------------------------------------------
    def _spatial_pillar(
        self,
        frames: list[np.ndarray],
        aspect_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[list[str]]]:
        cv2 = self._cv2()
        engineered_rows: list[np.ndarray] = []
        reliabilities: list[float] = []
        labels_all: list[list[str]] = []
        face_counts: list[float] = []
        complexity: list[float] = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayf = gray.astype(np.float32) / 255.0
            edges = cv2.Canny(gray, 70, 150)
            edge_density = float(np.mean(edges > 0))
            orientation_entropy = self._edge_orientation_entropy(grayf)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            texture = self._clip01(np.log1p(laplacian_var) / np.log1p(900.0))
            entropy = self._histogram_entropy(grayf, 32)
            height, width = gray.shape
            center = grayf[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
            center_edges = edges[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
            center_saliency = self._clip01(
                0.5 * (float(np.mean(center_edges > 0)) / (edge_density + 1e-6))
                + 0.5 * abs(float(np.mean(center)) - float(np.mean(grayf))) / 0.20
            )
            horizontal_balance = self._clip01(1.0 - abs(float(np.mean(grayf[:, : width // 2])) - float(np.mean(grayf[:, width // 2:]))) / 0.45)
            vertical_balance = self._clip01(1.0 - abs(float(np.mean(grayf[: height // 2, :])) - float(np.mean(grayf[height // 2 :, :]))) / 0.45)
            thirds = self._rule_of_thirds_proxy(edges)
            contour_density, largest_region = self._contour_statistics(edges)
            face_count, face_area = self._face_statistics(gray)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
            center_color_contrast = self._clip01(abs(float(np.mean(hsv[height // 4: 3 * height // 4, width // 4: 3 * width // 4, 1])) - float(np.mean(hsv[..., 1]))) / 0.25)
            vertical_lighting = float(np.clip((np.mean(grayf[: height // 2, :]) - np.mean(grayf[height // 2 :, :])) / 0.5, -1.0, 1.0))
            row = np.asarray(
                [
                    edge_density, orientation_entropy, texture, entropy, center_saliency,
                    horizontal_balance, vertical_balance, thirds, contour_density,
                    largest_region, min(face_count, 5.0) / 5.0, face_area,
                    float(np.clip((aspect_ratio - 1.0) / 1.5, -1.0, 1.0)),
                    vertical_lighting, center_color_contrast, 0.0,
                ],
                dtype=np.float32,
            )
            reliability = self._clip01(
                0.42 * min(1.0, edge_density / 0.08)
                + 0.32 * texture
                + 0.26 * entropy
            )
            labels: list[str] = []
            if face_count > 0:
                labels.append("face_visible_proxy")
            if edge_density > 0.11 or entropy > 0.78:
                labels.append("visually_complex_scene")
            elif edge_density < 0.025 and entropy < 0.45:
                labels.append("visually_simple_scene")
            if center_saliency > 0.68:
                labels.append("center_focused_composition_proxy")
            if horizontal_balance > 0.85 and vertical_balance > 0.85:
                labels.append("balanced_composition_proxy")
            if texture < 0.18:
                labels.append("low_spatial_detail")
            engineered_rows.append(row)
            reliabilities.append(reliability)
            labels_all.append(labels)
            face_counts.append(face_count)
            complexity.append(edge_density)

        engineered = np.asarray(engineered_rows, dtype=np.float32)
        context_projection, context_labels, context_used = self._pretrained_spatial_context(frames)
        if context_used:
            engineered[:, 15] = 1.0
            for index, label in enumerate(context_labels):
                if label:
                    labels_all[index].append(f"pretrained_context:{label}")
        if self.config.use_clip:
            features = context_projection
        else:
            features = np.concatenate([engineered, context_projection], axis=1).astype(np.float32)
        expected_dim = self.pillar_dims["spatial"]
        if features.shape != (len(frames), expected_dim):
            raise RuntimeError(
                f"Spatial feature contract produced {features.shape}; expected "
                f"({len(frames)}, {expected_dim})."
            )
        evidence = {
            "mean_edge_density": self._float(float(np.mean(complexity))),
            "mean_face_count_proxy": self._float(float(np.mean(face_counts))),
            "pretrained_context_status": self._spatial_model_status,
            "pretrained_context_used": bool(context_used),
            "note": (
                "Face and pretrained context entries are optional visual proxies. "
                "They do not establish identity, scene truth, or intent."
            ),
        }
        return features, np.asarray(reliabilities, dtype=np.float32), evidence, labels_all

    def _pretrained_spatial_context(
        self,
        frames: list[np.ndarray],
    ) -> tuple[np.ndarray, list[str], bool]:
        context_dim = CLIP_CONTEXT_DIM if self.config.use_clip else RESNET_CONTEXT_DIM
        empty = np.zeros((len(frames), context_dim), dtype=np.float32)
            
        model = self._maybe_load_spatial_model()
        if model is None:
            return empty, ["" for _ in frames], False
            
        try:
            torch = self._spatial_torch
            assert torch is not None
            
            if self.config.use_clip:
                # CLIP Extraction
                processor, clip_model = model
                device = clip_model.device
                
                # Frames are BGR from cv2, convert to RGB
                from PIL import Image
                pil_images = [Image.fromarray(frame[..., ::-1]) for frame in frames]
                
                inputs = processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                
                projected = image_features.detach().cpu().numpy().astype(np.float32)
                if projected.shape != (len(frames), CLIP_CONTEXT_DIM):
                    raise RuntimeError(
                        f"CLIP returned {projected.shape}; expected "
                        f"({len(frames)}, {CLIP_CONTEXT_DIM})."
                    )
                labels = ["clip_visual_context" for _ in frames]
                self._spatial_model_status = "cached_pretrained_clip"
                return projected, labels, True
            else:
                # ResNet18 Extraction
                rgb = np.stack([frame[..., ::-1] for frame in frames], axis=0).copy()
                tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
                tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
                mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(1, 3, 1, 1)
                std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(1, 3, 1, 1)
                device = next(model.parameters()).device
                tensor = ((tensor - mean) / std).to(device)
                with torch.no_grad():
                    embedding, logits = self._resnet18_forward(model, tensor, torch)
                projected = np.tanh(embedding.detach().cpu().numpy().astype(np.float32) @ self._resnet_projection)
                top_indices = logits.argmax(dim=1).detach().cpu().tolist()
                labels = [
                    self._spatial_categories[index] if 0 <= int(index) < len(self._spatial_categories) else ""
                    for index in top_indices
                ]
                self._spatial_model_status = "cached_pretrained_resnet18"
                return projected.astype(np.float32), labels, True
                
        except Exception as error:
            self._spatial_model_status = f"pretrained_context_failed:{type(error).__name__}"
            return empty, ["" for _ in frames], False

    @staticmethod
    def _resnet18_forward(model: Any, tensor: Any, torch: Any) -> tuple[Any, Any]:
        """Return penultimate ResNet-18 embedding and ImageNet logits."""

        x = model.conv1(tensor)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        embedding = torch.flatten(model.avgpool(x), 1)
        return embedding, model.fc(embedding)

    def _maybe_load_spatial_model(self) -> Any | None:
        if self._spatial_model_checked:
            return self._spatial_model
        self._spatial_model_checked = True
        if not self.config.use_pretrained_spatial:
            self._spatial_model_status = "disabled"
            return None
        try:
            import torch
            self._spatial_torch = torch
            device = torch.device(self.config.spatial_device)
            
            if self.config.use_clip:
                from transformers import CLIPModel, CLIPProcessor
                model_id = self.config.clip_model_id
                load_kwargs: dict[str, Any] = {"local_files_only": True}
                if self.config.clip_revision:
                    load_kwargs["revision"] = self.config.clip_revision
                # local_files_only is required: extraction may use an already
                # cached model but must never fetch model/config files while
                # building a dataset cache or serving an upload.
                processor = CLIPProcessor.from_pretrained(model_id, **load_kwargs)
                model = CLIPModel.from_pretrained(model_id, **load_kwargs).to(device).eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                self._spatial_model = (processor, model)
                self._spatial_model_status = "cached_pretrained_clip"
            else:
                from torchvision.models import ResNet18_Weights, resnet18
                weights = ResNet18_Weights.DEFAULT
                checkpoint_name = Path(str(weights.url)).name
                checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
                if not self.config.allow_model_download and not checkpoint_path.is_file():
                    self._spatial_model_status = "pretrained_weights_not_cached"
                    return None
                model = resnet18(weights=weights)
                model = model.to(device).eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                self._spatial_model = model
                categories = weights.meta.get("categories", [])
                self._spatial_categories = [str(item) for item in categories]
                self._spatial_model_status = "cached_pretrained_resnet18"
                
        except Exception as error:
            self._spatial_model = None
            self._spatial_model_status = f"pretrained_context_unavailable:{type(error).__name__}"
        return self._spatial_model

    # ------------------------------------------------------------------
    # Pillar 4: local optical-flow motion / kinematic context.
    # ------------------------------------------------------------------
    def _motion_pillar(
        self,
        frames: list[np.ndarray],
        motion_frames: list[np.ndarray],
        motion_deltas: np.ndarray,
        spatial_reliability: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[list[str]]]:
        cv2 = self._cv2()
        rows: list[np.ndarray] = []
        availability: list[float] = []
        reliability: list[float] = []
        labels_all: list[list[str]] = []
        mean_speeds: list[float] = []
        intensity: list[float] = []
        previous_speed = 0.0
        for index, (frame, paired, delta) in enumerate(zip(frames, motion_frames, motion_deltas.tolist())):
            # Identical high-detail frames are valid evidence of stillness;
            # only a missing pair is unavailable.  This preserves the useful
            # distinction between "no motion" and "no motion observation".
            if delta <= 0.0:
                rows.append(np.zeros(PILLAR_DIMS["motion"], dtype=np.float32))
                availability.append(0.0)
                reliability.append(0.0)
                labels_all.append(["motion_unavailable"])
                mean_speeds.append(0.0)
                intensity.append(0.0)
                continue
            source_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (self.config.flow_size, self.config.flow_size))
            paired_gray = cv2.resize(cv2.cvtColor(paired, cv2.COLOR_BGR2GRAY), (self.config.flow_size, self.config.flow_size))
            flow = cv2.calcOpticalFlowFarneback(
                source_gray,
                paired_gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitude = np.asarray(magnitude, dtype=np.float32)
            angle = np.asarray(angle, dtype=np.float32)
            raw_mean = float(np.mean(magnitude))
            raw_p90 = float(np.percentile(magnitude, 90))
            raw_std = float(np.std(magnitude))
            speed = self._clip01(raw_mean / 4.0)
            p90 = self._clip01(raw_p90 / 6.0)
            speed_std = self._clip01(raw_std / 4.0)
            active_fraction = float(np.mean(magnitude > 0.75))
            vector_mean = np.mean(flow.reshape(-1, 2), axis=0)
            global_magnitude = float(np.linalg.norm(vector_mean))
            coherence = self._clip01(global_magnitude / (raw_mean + 1e-6))
            weights = magnitude.reshape(-1)
            direction_hist, _ = np.histogram(angle.reshape(-1), bins=8, range=(0.0, 2.0 * np.pi), weights=weights)
            direction_hist = direction_hist.astype(np.float32)
            direction_hist /= direction_hist.sum() + 1e-8
            direction_entropy = self._clip01(float(-(direction_hist * np.log(direction_hist + 1e-8)).sum() / np.log(8.0)))
            flow_spread = self._clip01(float(np.percentile(magnitude, 90) - np.percentile(magnitude, 50)) / 5.0)
            localized_fraction = self._clip01(active_fraction * (1.0 - 0.5 * coherence))
            acceleration = self._clip01(abs(speed - previous_speed) / 0.35) if index else 0.0
            # This only measures high, abrupt kinematic activity.  It must not
            # be reported or trained as a violence/safety classifier.
            kinematic_intensity = self._clip01(0.55 * speed + 0.25 * acceleration + 0.20 * localized_fraction)
            row = np.asarray(
                [
                    speed, p90, speed_std, self._clip01(active_fraction), coherence,
                    direction_entropy, self._clip01(global_magnitude / 4.0), acceleration,
                    float(np.clip(vector_mean[0] / 4.0, -1.0, 1.0)),
                    float(np.clip(vector_mean[1] / 4.0, -1.0, 1.0)), localized_fraction,
                    flow_spread, kinematic_intensity, 1.0,
                ],
                dtype=np.float32,
            )
            gradient_quality = self._clip01(float(np.std(source_gray.astype(np.float32) / 255.0)) / 0.12)
            quality = self._clip01(0.65 * gradient_quality + 0.35 * float(spatial_reliability[index]))
            row_labels: list[str] = []
            if speed < 0.08:
                row_labels.append("still_or_slow_motion")
            elif speed > 0.42:
                row_labels.append("rapid_motion")
            else:
                row_labels.append("moderate_motion")
            if coherence > 0.70 and speed > 0.10:
                row_labels.append("coherent_global_motion_proxy")
            if acceleration > 0.50:
                row_labels.append("abrupt_motion_change")
            if kinematic_intensity > 0.55:
                row_labels.append("high_kinematic_intensity_proxy")
            rows.append(row)
            availability.append(1.0)
            reliability.append(quality)
            labels_all.append(row_labels)
            mean_speeds.append(raw_mean)
            intensity.append(kinematic_intensity)
            previous_speed = speed

        evidence = {
            "mean_flow_pixels": self._float(float(np.mean(mean_speeds))),
            "mean_kinematic_intensity_proxy": self._float(float(np.mean(intensity))),
            "availability_fraction": self._float(float(np.mean(availability))),
            "note": "Motion features quantify visual change; they do not detect violence, harm, or safety.",
        }
        return (
            np.asarray(rows, dtype=np.float32),
            np.asarray(availability, dtype=np.float32),
            np.asarray(reliability, dtype=np.float32),
            evidence,
            labels_all,
        )

    # ------------------------------------------------------------------
    # Pillar 5: temporal sequence summary.
    # ------------------------------------------------------------------
    def _temporal_pillar(
        self,
        timestamps: np.ndarray,
        color: np.ndarray,
        color_reliability: np.ndarray,
        spatial_reliability: np.ndarray,
        motion: np.ndarray,
        motion_available: np.ndarray,
        motion_reliability: np.ndarray,
        audio: np.ndarray,
        audio_available: np.ndarray,
        audio_reliability: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[list[str]]]:
        t = len(timestamps)
        if t == 0:
            raise ValueError("Temporal pillar received an empty sequence")
        rows = np.zeros((t, PILLAR_DIMS["temporal"]), dtype=np.float32)
        availability = np.zeros(t, dtype=np.float32)
        reliability = np.zeros(t, dtype=np.float32)
        labels_all: list[list[str]] = []
        cumulative_change = 0.0
        visual_changes: list[float] = []
        cut_like: list[float] = []
        for index in range(t):
            relative_time = float(index / max(t - 1, 1))
            if index == 0:
                luminance_change = 0.0
                color_change = 0.0
                brightness_delta = 0.0
                saturation_delta = 0.0
                motion_acceleration = 0.0
                audio_energy_change = 0.0
            else:
                luminance_change = self._clip01(abs(float(color[index, 0] - color[index - 1, 0])) / 0.35)
                # Hue circular components + saturation/brightness describe
                # colour evolution without assuming a fixed hue order.
                hue_change = float(np.sqrt(np.sum(np.square(color[index, 6:8] - color[index - 1, 6:8]))))
                color_change = self._clip01(0.45 * hue_change + 0.30 * abs(float(color[index, 4] - color[index - 1, 4])) + 0.25 * luminance_change)
                brightness_delta = float(np.clip((color[index, 0] - color[index - 1, 0]) / 0.35, -1.0, 1.0))
                saturation_delta = float(np.clip((color[index, 4] - color[index - 1, 4]) / 0.35, -1.0, 1.0))
                motion_acceleration = self._clip01(abs(float(motion[index, 0] - motion[index - 1, 0])) / 0.35)
                if audio_available[index] > 0.5 and audio_available[index - 1] > 0.5:
                    audio_energy_change = self._clip01(abs(float(audio[index, 0] - audio[index - 1, 0])) / 0.35)
                else:
                    audio_energy_change = 0.0
            visual_activity = self._clip01(0.50 * luminance_change + 0.30 * color_change + 0.20 * float(motion[index, 0]))
            cumulative_change = self._clip01(cumulative_change + visual_activity / max(t - 1, 1))
            cut_proxy = self._clip01((0.65 * luminance_change + 0.35 * color_change - 0.16) / 0.42)
            audio_visual_sync = 0.0
            if audio_available[index] > 0.5 and motion_available[index] > 0.5:
                audio_visual_sync = self._clip01(1.0 - abs(float(audio[index, 9]) - float(motion[index, 0])))
            rows[index] = np.asarray(
                [
                    relative_time, luminance_change, color_change, brightness_delta,
                    saturation_delta, cut_proxy, float(motion[index, 0]), motion_acceleration,
                    visual_activity, cumulative_change, audio_energy_change, audio_visual_sync,
                ],
                dtype=np.float32,
            )
            if t > 1:
                availability[index] = 1.0
                base_quality = 0.40 * color_reliability[index] + 0.40 * spatial_reliability[index] + 0.20 * max(float(motion_reliability[index]), 0.25)
                reliability[index] = self._clip01(base_quality)
            labels: list[str] = []
            if index == 0:
                labels.append("sequence_start")
            elif visual_activity < 0.10:
                labels.append("visually_stable_interval")
            elif visual_activity > 0.48:
                labels.append("rapid_visual_change")
            else:
                labels.append("gradual_visual_change")
            if cut_proxy > 0.55:
                labels.append("cut_like_transition_proxy")
            if brightness_delta > 0.22:
                labels.append("brightening")
            elif brightness_delta < -0.22:
                labels.append("darkening")
            if motion_acceleration > 0.50:
                labels.append("motion_accelerating")
            labels_all.append(labels)
            visual_changes.append(visual_activity)
            cut_like.append(cut_proxy)

        evidence = {
            "mean_visual_activity": self._float(float(np.mean(visual_changes))),
            "cut_like_fraction": self._float(float(np.mean(np.asarray(cut_like) > 0.55))),
            "sequence_duration_seconds": self._float(float(timestamps[-1] - timestamps[0])) if t > 1 else 0.0,
            "note": "Temporal cues summarize visual evolution and pacing, not a sentiment trajectory by themselves.",
        }
        return rows, availability, reliability, evidence, labels_all

    # ------------------------------------------------------------------
    # Spatial / image statistics helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _colorfulness(frame: np.ndarray) -> float:
        blue, green, red = [channel.astype(np.float32) for channel in np.dsplit(frame, 3)]
        red_green = red - green
        yellow_blue = 0.5 * (red + green) - blue
        return float(
            np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
            + 0.3 * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
        )

    @staticmethod
    def _histogram_entropy(image: np.ndarray, bins: int) -> float:
        histogram, _ = np.histogram(image.reshape(-1), bins=bins, range=(0.0, 1.0))
        probabilities = histogram.astype(np.float64) / max(float(histogram.sum()), 1.0)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        return float(np.clip(entropy / np.log2(float(bins)), 0.0, 1.0))

    def _edge_orientation_entropy(self, gray: np.ndarray) -> float:
        cv2 = self._cv2()
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y)
        histogram, _ = np.histogram(angle.reshape(-1), bins=8, range=(0.0, 2.0 * np.pi), weights=magnitude.reshape(-1))
        probabilities = histogram.astype(np.float64)
        probabilities /= probabilities.sum() + 1e-12
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(8.0)
        return self._clip01(float(entropy))

    def _rule_of_thirds_proxy(self, edges: np.ndarray) -> float:
        height, width = edges.shape
        band = max(1, min(height, width) // 24)
        positions_x = (width // 3, (2 * width) // 3)
        positions_y = (height // 3, (2 * height) // 3)
        mask = np.zeros_like(edges, dtype=bool)
        for x in positions_x:
            mask[:, max(0, x - band): min(width, x + band + 1)] = True
        for y in positions_y:
            mask[max(0, y - band): min(height, y + band + 1), :] = True
        local_density = float(np.mean(edges[mask] > 0)) if np.any(mask) else 0.0
        global_density = float(np.mean(edges > 0))
        return self._clip01(local_density / (2.0 * global_density + 1e-6))

    @staticmethod
    def _contour_statistics(edges: np.ndarray) -> tuple[float, float]:
        cv2 = SemanticPillarExtractor._cv2()
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = float(edges.shape[0] * edges.shape[1])
        if not contours:
            return 0.0, 0.0
        areas = [float(cv2.contourArea(contour)) for contour in contours]
        contour_density = min(len(contours) / 120.0, 1.0)
        largest_fraction = min(max(areas) / max(area, 1.0), 1.0)
        return float(contour_density), float(largest_fraction)

    def _face_statistics(self, gray: np.ndarray) -> tuple[float, float]:
        detector = self._maybe_load_face_detector()
        if detector is None:
            return 0.0, 0.0
        try:
            faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(20, 20))
            if faces is None or len(faces) == 0:
                return 0.0, 0.0
            frame_area = max(float(gray.shape[0] * gray.shape[1]), 1.0)
            largest_fraction = max(float(width * height) / frame_area for _, _, width, height in faces)
            return float(len(faces)), self._clip01(largest_fraction * 5.0)
        except Exception:
            return 0.0, 0.0

    def _maybe_load_face_detector(self) -> Any | None:
        if self._face_detector_checked:
            return self._face_detector
        self._face_detector_checked = True
        try:
            cv2 = self._cv2()
            cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(str(cascade))
            if detector.empty():
                return None
            self._face_detector = detector
        except Exception:
            self._face_detector = None
        return self._face_detector

    # ------------------------------------------------------------------
    # Dependency and output helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _cv2() -> Any:
        try:
            import cv2
        except Exception as error:  # pragma: no cover - depends on runtime
            raise RuntimeError("opencv-python is required for video semantic extraction") from error
        return cv2

    @staticmethod
    def _librosa() -> Any | None:
        try:
            import librosa
        except Exception:
            return None
        return librosa

    def _validate_output(
        self,
        features: dict[str, np.ndarray],
        availability: dict[str, np.ndarray],
        reliability: dict[str, np.ndarray],
    ) -> None:
        lengths: set[int] = set()
        expected_dims = self.pillar_dims
        for name in PILLAR_NAMES:
            values = features[name]
            expected_dim = expected_dims[name]
            if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != expected_dim:
                raise RuntimeError(
                    f"Invalid {name} feature array; expected float32 (T, {expected_dim}), got "
                    f"{values.dtype} {values.shape}."
                )
            lengths.add(int(values.shape[0]))
            for kind, values_1d in (("availability", availability[name]), ("reliability", reliability[name])):
                if values_1d.dtype != np.float32 or values_1d.ndim != 1 or values_1d.shape[0] != values.shape[0]:
                    raise RuntimeError(f"Invalid {kind} array for {name}")
                if not np.all(np.isfinite(values_1d)) or np.any(values_1d < 0.0) or np.any(values_1d > 1.0):
                    raise RuntimeError(f"{kind} for {name} must be finite and in [0, 1]")
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"{name} features contain non-finite values")
        if len(lengths) != 1:
            raise RuntimeError("All semantic pillars must share a sequence length")

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _float(value: float) -> float:
        return float(round(float(value), 6))


# A short alias makes the intended import discoverable without forcing callers
# to know whether a project calls these "semantic" or "five pillar" features.
FivePillarSemanticExtractor = SemanticPillarExtractor


__all__ = [
    "SCHEMA_VERSION",
    "LEGACY_BASELINE_SCHEMA_VERSION",
    "EXTRACTOR_IMPLEMENTATION_VERSION",
    "PILLAR_NAMES",
    "PILLAR_DIMS",
    "PILLAR_FEATURES",
    "SemanticPillarConfig",
    "SemanticPillarExtractor",
    "FivePillarSemanticExtractor",
]
