"""Transparent inference for the canonical semantic five-pillar model.

This module is intentionally separate from the legacy raw-frame inference
path.  It loads only a checkpoint that declares the semantic LIRIS task,
recreates the exact feature-normalisation contract saved during training, and
then analyses one video of at most 15 seconds.

The report distinguishes an unavailable modality from weak evidence.  Thus a
mute upload has an ``audio`` pillar of ``NIL`` and an exactly-zero audio fusion
weight, whereas a black or low-detail visual upload retains its visual pillars
with low reliability.  Neither condition is itself treated as a sentiment
label.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:  # Package launch: ``python -m scene_motion_llm...``
    from scene_motion_llm.models.semantic_five_pillar import (
        CLASS_NAMES,
        PILLAR_NAMES,
        SemanticFivePillarConfig,
        SemanticFivePillarModel,
    )
    from scene_motion_llm.utils.semantic_pillars import (
        LEGACY_BASELINE_SCHEMA_VERSION,
        PILLAR_DIMS,
        SCHEMA_VERSION,
        SemanticPillarConfig,
        SemanticPillarExtractor,
    )
except ModuleNotFoundError as error:  # Direct ``streamlit run app.py`` launch
    if error.name != "scene_motion_llm":
        raise
    from models.semantic_five_pillar import (
        CLASS_NAMES,
        PILLAR_NAMES,
        SemanticFivePillarConfig,
        SemanticFivePillarModel,
    )
    from utils.semantic_pillars import (
        LEGACY_BASELINE_SCHEMA_VERSION,
        PILLAR_DIMS,
        SCHEMA_VERSION,
        SemanticPillarConfig,
        SemanticPillarExtractor,
    )


CANONICAL_TASK_NAMES = frozenset(
    {
        "liris_semantic_five_pillar",
        "semantic_five_pillar",
    }
)


class CheckpointContractError(ValueError):
    """Raised when a checkpoint cannot safely drive semantic inference."""


def _json_safe(value: Any) -> Any:
    """Recursively turn NumPy and Torch scalars into plain JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointContractError(f"Checkpoint field '{name}' must be a mapping.")
    return value


def _checkpoint_mapping(checkpoint: Mapping[str, Any], *names: str) -> Mapping[str, Any] | None:
    for name in names:
        value = checkpoint.get(name)
        if isinstance(value, Mapping):
            return value
    return None


def _safe_torch_load(path: Path) -> Mapping[str, Any]:
    """Read tensor/primitive checkpoint data without unpickling arbitrary code."""

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise CheckpointContractError(
            "Semantic checkpoint loading requires a PyTorch release with "
            "torch.load(..., weights_only=True)."
        ) from error
    except Exception as error:
        raise CheckpointContractError(
            f"Could not safely load semantic checkpoint '{path}': {type(error).__name__}: {error}"
        ) from error
    return _mapping(checkpoint, name="root")


def _model_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_input_dims: Mapping[str, int],
) -> SemanticFivePillarConfig:
    raw = _checkpoint_mapping(checkpoint, "model_config", "semantic_model_config")
    if raw is None:
        raise CheckpointContractError(
            "The semantic checkpoint is missing 'model_config'; architecture reconstruction "
            "must not rely on defaults."
        )
    allowed = {field.name for field in fields(SemanticFivePillarConfig)}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise CheckpointContractError(
            f"Checkpoint model_config contains unsupported keys: {sorted(unknown)}."
        )
    values = dict(raw)
    if "input_dims" not in values:
        raise CheckpointContractError("Checkpoint model_config is missing 'input_dims'.")
    values["input_dims"] = dict(_mapping(values["input_dims"], name="model_config.input_dims"))
    if "class_names" in values:
        values["class_names"] = tuple(values["class_names"])
    try:
        config = SemanticFivePillarConfig(**values)
    except (TypeError, ValueError) as error:
        raise CheckpointContractError(f"Invalid semantic model_config: {error}") from error
    if tuple(config.class_names) != CLASS_NAMES:
        raise CheckpointContractError(
            "Semantic checkpoint class order must be ('Positive', 'Neutral', 'Negative')."
        )
    if dict(config.input_dims) != dict(expected_input_dims):
        raise CheckpointContractError(
            "Checkpoint input dimensions do not match the installed semantic extractor: "
            f"checkpoint={dict(config.input_dims)}, extractor={dict(expected_input_dims)}."
        )
    return config


def _extractor_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> SemanticPillarConfig:
    raw = _checkpoint_mapping(checkpoint, "extractor_config", "semantic_extractor_config")
    if raw is None:
        nested = _checkpoint_mapping(checkpoint, "extractor")
        if nested is not None:
            candidate = nested.get("config")
            if isinstance(candidate, Mapping):
                raw = candidate
    if raw is None:
        raise CheckpointContractError(
            "The semantic checkpoint is missing 'extractor_config'; training and inference "
            "must use the same semantic extraction settings."
        )

    # Some training metadata stores ``config_metadata()`` rather than the
    # config itself.  Peel that harmless wrapper while rejecting ambiguity.
    if isinstance(raw.get("config"), Mapping):
        raw = _mapping(raw["config"], name="extractor_config.config")
    allowed = {field.name for field in fields(SemanticPillarConfig)}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise CheckpointContractError(
            f"Checkpoint extractor_config contains unsupported keys: {sorted(unknown)}."
        )
    try:
        values = dict(raw)
        configured_limit = float(values.get("max_duration_seconds", 15.0))
        if configured_limit > 15.0 + 1e-6:
            raise CheckpointContractError(
                "This checkpoint was configured for clips longer than 15 seconds and "
                "is not valid for the canonical short-video interface."
            )
        # Inference must never trigger a network download.  A pretrained
        # spatial model may still be used when its weights are already cached.
        values["allow_model_download"] = False
        if values.get("spatial_device") == "cuda" and not torch.cuda.is_available():
            values["spatial_device"] = "cpu"
        config = SemanticPillarConfig(**values)
    except CheckpointContractError:
        raise
    except (TypeError, ValueError) as error:
        raise CheckpointContractError(f"Invalid semantic extractor_config: {error}") from error
    if config.max_duration_seconds > 15.0 + 1e-6:
        raise CheckpointContractError("Semantic inference accepts videos of at most 15 seconds.")
    return config


def _normalizer_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_input_dims: Mapping[str, int],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load train-set mean/std vectors without silently inventing statistics.

    The canonical trainer writes ``normalizer`` as either
    ``{pillar: {mean, std}}`` or ``{means: {pillar: ...}, stds: {pillar: ...}}``.
    The latter spelling is accepted to make report/inference code stable across
    small trainer refactors, but an absent normalizer is always an error.
    """

    raw = _checkpoint_mapping(checkpoint, "normalizer", "feature_normalizer", "normalization")
    if raw is None:
        raise CheckpointContractError(
            "The semantic checkpoint is missing train-set normalizer statistics; "
            "refusing to normalize uploads with ad-hoc per-video statistics."
        )
    # The canonical trainer nests the per-pillar vectors under ``pillars`` so
    # it can record normalizer format/epsilon too.  Accept that state directly
    # while retaining the flat spellings for previously saved experiments.
    if isinstance(raw.get("pillars"), Mapping):
        raw = _mapping(raw["pillars"], name="normalizer.pillars")
    means = raw.get("means") if isinstance(raw.get("means"), Mapping) else None
    stds = raw.get("stds") if isinstance(raw.get("stds"), Mapping) else None
    if means is None or stds is None:
        means = raw.get("mean") if isinstance(raw.get("mean"), Mapping) else means
        stds = raw.get("std") if isinstance(raw.get("std"), Mapping) else stds

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pillar in PILLAR_NAMES:
        if means is not None and stds is not None:
            raw_mean, raw_std = means.get(pillar), stds.get(pillar)
        else:
            entry = raw.get(pillar)
            if not isinstance(entry, Mapping):
                raise CheckpointContractError(
                    f"Normalizer is missing statistics for pillar '{pillar}'."
                )
            raw_mean, raw_std = entry.get("mean"), entry.get("std")
        if raw_mean is None or raw_std is None:
            raise CheckpointContractError(
                f"Normalizer is missing mean/std values for pillar '{pillar}'."
            )
        mean = np.asarray(_json_safe(raw_mean), dtype=np.float32).reshape(-1)
        std = np.asarray(_json_safe(raw_std), dtype=np.float32).reshape(-1)
        expected_dim = int(expected_input_dims[pillar])
        if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
            raise CheckpointContractError(
                f"Normalizer '{pillar}' has mean/std shapes {mean.shape}/{std.shape}; "
                f"expected ({expected_dim},)."
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise CheckpointContractError(
                f"Normalizer '{pillar}' contains non-finite or non-positive standard deviations."
            )
        result[pillar] = (mean, std)
    return result


class SemanticFivePillarInferencer:
    """Run transparent P/N/N inference for one short video.

    Args:
        checkpoint_path: A canonical semantic five-pillar checkpoint produced
            from LIRIS-ACCEDE.  Legacy raw-frame and feature-file checkpoints
            are rejected deliberately.
        device: ``"cpu"``, ``"cuda"``, or ``None`` for the safest available
            compute device.  Feature extraction remains CPU-first unless a
            cached spatial encoder was deliberately configured at training.

    The returned dictionary is JSON-safe and contains the full class
    probability distribution, fusion weights, per-pillar availability and
    reliability, and descriptive extraction cues.
    """

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Semantic checkpoint was not found: {self.checkpoint_path}")

        checkpoint = _safe_torch_load(self.checkpoint_path)
        task = checkpoint.get("task")
        if task not in CANONICAL_TASK_NAMES:
            raise CheckpointContractError(
                "This is not a canonical semantic five-pillar checkpoint. "
                f"Expected one of {sorted(CANONICAL_TASK_NAMES)}, got {task!r}."
            )
        self.extractor_config = _extractor_config_from_checkpoint(checkpoint)
        # The feature widths are configuration-dependent.  Build the exact
        # extractor contract before accepting a model or normalizer so a
        # PANNs/CLIP checkpoint cannot be interpreted as the 20/24-D baseline.
        self.expected_input_dims = SemanticPillarExtractor(self.extractor_config).pillar_dims
        self.model_config = _model_config_from_checkpoint(
            checkpoint,
            expected_input_dims=self.expected_input_dims,
        )
        checkpoint_schema = checkpoint.get("extractor_schema_version", checkpoint.get("schema_version"))
        self._validate_checkpoint_schema(
            checkpoint_schema,
            extractor_config=self.extractor_config,
            input_dims=self.model_config.input_dims,
        )
        self.checkpoint_schema_version = str(checkpoint_schema)
        self.normalizer = _normalizer_from_checkpoint(
            checkpoint,
            expected_input_dims=self.expected_input_dims,
        )
        self.device = self._resolve_device(device)
        self.model = SemanticFivePillarModel(self.model_config).to(self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if not isinstance(state_dict, Mapping):
            raise CheckpointContractError("Checkpoint is missing a mapping 'model_state_dict'.")
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise CheckpointContractError(
                "Semantic checkpoint weights do not match its declared architecture."
            ) from error
        self.model.eval()
        self.extractor = SemanticPillarExtractor(self.extractor_config)
        self.class_names = tuple(self.model.class_names)
        self.checkpoint_metadata = {
            "task": task,
            "path": str(self.checkpoint_path),
            "extractor_schema_version": SCHEMA_VERSION,
            "checkpoint_extractor_schema_version": self.checkpoint_schema_version,
            "input_dims": dict(self.expected_input_dims),
            "model_config": _json_safe(asdict(self.model_config)),
            "extractor_config": _json_safe(asdict(self.extractor_config)),
            "validation_metrics": _json_safe(checkpoint.get("validation_metrics", {})),
        }

    @staticmethod
    def _validate_checkpoint_schema(
        checkpoint_schema: Any,
        *,
        extractor_config: SemanticPillarConfig,
        input_dims: Mapping[str, int],
    ) -> None:
        """Accept only the current contract or a safe unchanged v2 baseline.

        v2 optional PANNs/CLIP extraction had incompatible cache and tensor
        dimensions, so those checkpoints are deliberately not recoverable.
        The existing v2 baseline remains usable because it has the exact
        20/16/24/14/12 feature ordering and widths used by v3's baseline.
        """

        if checkpoint_schema == SCHEMA_VERSION:
            return
        if checkpoint_schema == LEGACY_BASELINE_SCHEMA_VERSION:
            is_baseline = (
                not extractor_config.use_panns
                and not extractor_config.use_clip
                and dict(input_dims) == dict(PILLAR_DIMS)
            )
            if is_baseline:
                return
            raise CheckpointContractError(
                "A v2 checkpoint is compatible only with the unchanged baseline feature "
                "contract. Re-extract and retrain optional PANNs/CLIP experiments under v3."
            )
        raise CheckpointContractError(
            f"Checkpoint extractor schema is {checkpoint_schema!r}; installed schema is "
            f"{SCHEMA_VERSION!r}. Re-extract/retrain rather than mixing schemas."
        )

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for semantic inference but is not available.")
        return resolved

    def _prepare_model_inputs(
        self, extraction: Mapping[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
        if extraction.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Extractor returned schema {extraction.get('schema_version')!r}, expected {SCHEMA_VERSION!r}."
            )
        features = _mapping(extraction.get("features"), name="extraction.features")
        availability = _mapping(extraction.get("availability"), name="extraction.availability")
        reliability = _mapping(extraction.get("reliability"), name="extraction.reliability")
        extractor_metadata = extraction.get("extractor")
        if isinstance(extractor_metadata, Mapping):
            metadata_config = extractor_metadata.get("config")
            if isinstance(metadata_config, Mapping):
                declared_dims = metadata_config.get("pillar_dims")
                if isinstance(declared_dims, Mapping):
                    try:
                        normalized_declared_dims = {
                            pillar: int(declared_dims[pillar]) for pillar in PILLAR_NAMES
                        }
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError("Extractor metadata has invalid pillar_dims.") from error
                    if normalized_declared_dims != dict(self.expected_input_dims):
                        raise ValueError(
                            "Extractor metadata dimensions do not match this checkpoint: "
                            f"extraction={normalized_declared_dims}, "
                            f"checkpoint={dict(self.expected_input_dims)}."
                        )
        sequences: dict[str, torch.Tensor] = {}
        availability_tensors: dict[str, torch.Tensor] = {}
        reliability_tensors: dict[str, torch.Tensor] = {}
        sequence_length: int | None = None

        for pillar in PILLAR_NAMES:
            expected_dim = int(self.expected_input_dims[pillar])
            value = np.asarray(features.get(pillar), dtype=np.float32)
            mask = np.asarray(availability.get(pillar), dtype=np.float32).reshape(-1)
            quality = np.asarray(reliability.get(pillar), dtype=np.float32).reshape(-1)
            if value.ndim != 2 or value.shape[1] != expected_dim:
                raise ValueError(
                    f"Extractor '{pillar}' features must have shape (T, {expected_dim}); got {value.shape}."
                )
            if sequence_length is None:
                sequence_length = int(value.shape[0])
            if value.shape[0] != sequence_length or mask.shape != (sequence_length,) or quality.shape != (sequence_length,):
                raise ValueError(f"Extractor produced inconsistent sequence lengths for '{pillar}'.")

            value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            mask = np.clip(np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            quality = np.clip(np.nan_to_num(quality, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            mean, std = self.normalizer[pillar]
            standardized = (value - mean[None, :]) / std[None, :]
            # Do not turn a missing/mute zero vector into a nonzero normalized
            # vector.  The model also receives an explicit mask, but this
            # prevents encoder biases from seeing fabricated input values.
            standardized = np.where(mask[:, None] > 0.0, standardized, 0.0).astype(np.float32)
            sequences[pillar] = torch.from_numpy(standardized).unsqueeze(0).to(self.device)
            availability_tensors[pillar] = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            reliability_tensors[pillar] = torch.from_numpy(quality).unsqueeze(0).to(self.device)

        assert sequence_length is not None
        if sequence_length < 1:
            raise ValueError("Semantic extractor returned an empty video sequence.")
        return sequences, availability_tensors, reliability_tensors, sequence_length

    @staticmethod
    def _cue_summary(labels: Any, *, limit: int = 6) -> list[str]:
        """Return stable descriptive cue counts without treating them as labels."""

        counts: dict[str, int] = {}
        if isinstance(labels, list):
            for window in labels:
                if isinstance(window, list):
                    for item in window:
                        if isinstance(item, str) and item:
                            counts[item] = counts.get(item, 0) + 1
        return [
            cue
            for cue, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        ]

    @staticmethod
    def _mean_float(values: torch.Tensor) -> float:
        return float(values.detach().float().mean().cpu().item()) if values.numel() else 0.0

    @staticmethod
    def _format_percentage(value: float) -> str:
        return f"{100.0 * value:.1f}%"

    def _build_report(
        self,
        extraction: Mapping[str, Any],
        output: Mapping[str, Any],
        sequence_length: int,
    ) -> dict[str, Any]:
        probabilities = output["probabilities"].detach().cpu()[0]
        logits = output["logits"].detach().cpu()[0]
        if not torch.isfinite(probabilities).all() or not torch.isfinite(logits).all():
            raise RuntimeError("Semantic model returned non-finite probabilities or logits.")
        predicted_index = int(output["predicted_class"].detach().cpu()[0].item())
        sentiment = self.class_names[predicted_index]
        confidence = float(probabilities[predicted_index].item())

        extraction_evidence = _mapping(extraction.get("evidence"), name="extraction.evidence")
        semantic_labels = _mapping(extraction.get("semantic_labels"), name="extraction.semantic_labels")
        raw_availability = _mapping(extraction.get("availability"), name="extraction.availability")
        raw_reliability = _mapping(extraction.get("reliability"), name="extraction.reliability")
        fusion_weights = output["fusion_weights"].detach().cpu()[0]
        pillar_reliability = output["pillar_reliability"].detach().cpu()[0]
        fusion_mask = output["fusion_mask"].detach().cpu()[0]
        effective_reliability = _mapping(output["effective_reliability"], name="model.effective_reliability")
        if not torch.isfinite(fusion_weights).all():
            raise RuntimeError("Semantic model returned non-finite fusion weights.")

        pillars: dict[str, dict[str, Any]] = {}
        weights: dict[str, float] = {}
        warnings: list[str] = []
        for index, pillar in enumerate(PILLAR_NAMES):
            mask = np.asarray(raw_availability[pillar], dtype=np.float32)
            quality = np.asarray(raw_reliability[pillar], dtype=np.float32)
            effective = effective_reliability[pillar][0]
            available = bool(fusion_mask[index].item())
            fusion_weight = float(fusion_weights[index].item()) if available else 0.0
            # A zero-reliability audio sequence is a strict, testable contract:
            # it must never influence the fusion output.
            if pillar == "audio" and float(effective.sum().detach().cpu().item()) == 0.0:
                if abs(fusion_weight) > 1e-8:
                    raise RuntimeError(
                        "Mute/unavailable audio received nonzero fusion weight; checkpoint/model contract violated."
                    )
                fusion_weight = 0.0
            weights[pillar] = fusion_weight
            availability_fraction = float(np.clip(mask, 0.0, 1.0).mean()) if mask.size else 0.0
            raw_quality = float(np.clip(quality, 0.0, 1.0).mean()) if quality.size else 0.0
            status = "available" if available else "NIL"
            if pillar == "audio":
                audio_evidence = extraction_evidence.get("audio", {})
                if isinstance(audio_evidence, Mapping):
                    audio_status = str(audio_evidence.get("audio_status", "unavailable"))
                    status = "available" if available else "NIL"
                else:
                    audio_status = "unavailable"
            else:
                audio_status = None
            entry: dict[str, Any] = {
                "status": status,
                "available": available,
                "availability_fraction": availability_fraction,
                "mean_extractor_reliability": raw_quality,
                "mean_effective_reliability": float(pillar_reliability[index].item()),
                "fusion_weight": fusion_weight,
                "descriptive_cues": self._cue_summary(semantic_labels.get(pillar)),
                "evidence": _json_safe(extraction_evidence.get(pillar, {})),
            }
            if pillar == "audio":
                entry["audio_status"] = audio_status
            pillars[pillar] = entry

            if pillar in {"color", "spatial"} and available and raw_quality < 0.15:
                warnings.append(
                    f"{pillar.capitalize()} evidence is low-detail/low-information; this does not imply Negative sentiment."
                )
        if not pillars["audio"]["available"]:
            warnings.append(
                "Audio is NIL, so the prediction uses the remaining available pillars; no audio cue was fabricated."
            )
        if not any(pillars[name]["available"] for name in PILLAR_NAMES):
            warnings.append("No pillar contained usable evidence; treat the prediction as invalid.")

        temporal_weights = output["temporal_attention_weights"].detach().cpu()[0]
        timestamps = extraction.get("temporal_inputs", {}).get("timestamps_seconds", [])
        timestamps_array = np.asarray(timestamps, dtype=np.float32).reshape(-1)
        ranked_windows = torch.argsort(temporal_weights, descending=True).tolist()
        salient_windows: list[dict[str, Any]] = []
        for window_index in ranked_windows[: min(3, sequence_length)]:
            weight = float(temporal_weights[window_index].item())
            if weight <= 0.0:
                continue
            time_seconds = (
                float(timestamps_array[window_index])
                if window_index < len(timestamps_array)
                else None
            )
            salient_windows.append(
                {
                    "window_index": int(window_index),
                    "timestamp_seconds": time_seconds,
                    "temporal_attention_weight": weight,
                }
            )

        probability_map = {
            class_name: float(probabilities[index].item())
            for index, class_name in enumerate(self.class_names)
        }
        top_alternatives = sorted(probability_map.items(), key=lambda item: item[1], reverse=True)
        runner_up_name, runner_up_probability = top_alternatives[1]
        human_report = self._human_report(
            sentiment=sentiment,
            confidence=confidence,
            probabilities=probability_map,
            pillars=pillars,
            runner_up_name=runner_up_name,
            runner_up_probability=runner_up_probability,
        )
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probability_map,
            # Keep the small, stable keys that UI/API callers historically
            # consume, while retaining the richer transparent breakdown below.
            "evidence": _json_safe(extraction_evidence),
            "pillar_summary": pillars,
            "pillar_weights": weights,
            "audio_status": pillars["audio"].get("audio_status", "unavailable"),
            "prediction": {
                "class_index": predicted_index,
                "class_name": sentiment,
                "confidence": confidence,
                "logits": [float(value) for value in logits.tolist()],
            },
            "video": _json_safe(extraction.get("video", {})),
            "sequence_length": sequence_length,
            "fusion_weights": weights,
            "pillars": pillars,
            "salient_temporal_windows": salient_windows,
            "warnings": warnings,
            "extractor_notes": _json_safe(extraction_evidence.get("notes", {})),
            "checkpoint": self.checkpoint_metadata,
            "report": human_report,
        }

    def _human_report(
        self,
        *,
        sentiment: str,
        confidence: float,
        probabilities: Mapping[str, float],
        pillars: Mapping[str, Mapping[str, Any]],
        runner_up_name: str,
        runner_up_probability: float,
    ) -> str:
        lines = [
            "SEMANTIC FIVE-PILLAR VIDEO ANALYSIS",
            f"Prediction: {sentiment} ({self._format_percentage(confidence)} confidence)",
            "Class probabilities: " + ", ".join(
                f"{name} {self._format_percentage(value)}" for name, value in probabilities.items()
            ),
            (
                f"Closest alternative: {runner_up_name} "
                f"({self._format_percentage(runner_up_probability)})."
            ),
            "",
            "Pillar evidence (fusion weights are learned, reliability-aware weights):",
        ]
        for pillar in PILLAR_NAMES:
            entry = pillars[pillar]
            if entry["status"] == "NIL":
                suffix = ""
                if pillar == "audio":
                    suffix = f"; audio status: {entry.get('audio_status', 'unavailable')}"
                lines.append(f"- {pillar.capitalize()}: NIL (fusion 0.0%{suffix})")
                continue
            cues = ", ".join(entry["descriptive_cues"][:3]) or "no compact cue summary"
            lines.append(
                f"- {pillar.capitalize()}: fusion {self._format_percentage(float(entry['fusion_weight']))}; "
                f"reliability {self._format_percentage(float(entry['mean_effective_reliability']))}; "
                f"cues: {cues}."
            )
        lines.extend(
            [
                "",
                "Interpretation: cues are measured descriptions, not emotion or intent claims. "
                "Low light, a muted/absent soundtrack, low detail, or rapid motion are not hard-coded "
                "as Positive or Negative; the trained fusion model makes the P/N/N prediction from the "
                "available evidence.",
            ]
        )
        return "\n".join(lines)

    @torch.inference_mode()
    def analyze_video(self, video_path: str | Path) -> dict[str, Any]:
        """Analyse one decodable video of no more than 15 seconds.

        ``SemanticPillarExtractor`` enforces the duration before inference.  A
        mute/no-audio clip is valid and produces NIL audio rather than an
        exception; a frame decoder failure remains an exception because no
        visual evidence can be extracted.
        """

        extraction = self.extractor.extract(video_path)
        video = _mapping(extraction.get("video"), name="extraction.video")
        duration = float(video.get("duration_seconds", 0.0))
        if duration > 15.0 + 1e-6:
            raise ValueError(
                f"Video is {duration:.2f}s; canonical semantic inference accepts clips up to 15 seconds."
            )
        sequences, availability, reliability, sequence_length = self._prepare_model_inputs(extraction)
        output = self.model(sequences, availability, reliability)
        return self._build_report(extraction, output, sequence_length)


__all__ = [
    "CANONICAL_TASK_NAMES",
    "CheckpointContractError",
    "SemanticFivePillarInferencer",
]
