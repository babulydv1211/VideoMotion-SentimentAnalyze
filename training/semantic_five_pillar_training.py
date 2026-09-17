"""Reproducible LIRIS-ACCEDE training for semantic five-pillar fusion.

This module is intentionally the *canonical* raw-video training path for the
semantic five-pillar model.  It does not use the project's legacy random
splits or threshold labels.  Instead it uses LIRIS-ACCEDE's official split
file and fixed global thirds of the official ``valenceRank`` target:

* ranks 0--3265  -> Negative (class 2)
* ranks 3266--6532 -> Neutral (class 1)
* ranks 6533--9799 -> Positive (class 0)

The class order is always ``(Positive, Neutral, Negative)``.  The extractor
measurements are cached with a source fingerprint and extractor signature, so
a later preprocessing change cannot quietly reuse stale arrays.  Audio status
and decode errors are retained in cache metadata; a missing soundtrack is not
silently presented as a successful audio observation.

The public APIs deliberately require explicit dataset paths.  No path under a
user's Downloads or Desktop directory is embedded here, which keeps training
commands portable and makes a paper experiment reproducible.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing as multiprocessing
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid
import zipfile

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from scene_motion_llm.models.semantic_five_pillar import (
    CLASS_NAMES,
    PILLAR_NAMES,
    SemanticFivePillarConfig,
    SemanticFivePillarModel,
)
from scene_motion_llm.utils.semantic_pillars import (
    SCHEMA_VERSION,
    SemanticPillarConfig,
    SemanticPillarExtractor,
)


TOTAL_LIRIS_RANKS = 9_800
"""The official LIRIS-ACCEDE rank range is ``0..9799``."""

CACHE_FORMAT_VERSION = "semantic-five-pillar-cache-v1"
CHECKPOINT_TASK = "liris_semantic_five_pillar"
LABEL_RULE = (
    "Fixed global LIRIS-ACCEDE valenceRank thirds: ranks 0-3265=Negative "
    "(class 2), 3266-6532=Neutral (class 1), 6533-9799=Positive (class 0)."
)
OFFICIAL_SPLIT_CODES: Mapping[str, str] = {
    "1": "train",
    "2": "validation",
    "0": "test",
}

# The raw model module deliberately keeps generous architecture defaults for
# callers that have substantially more training data.  LIRIS's official train
# partition in this project has only 2,450 clips, so the canonical *training*
# entry point uses this compact regularised configuration unless a caller
# supplies an explicit ``model_config``.
DEFAULT_COMPACT_MODEL_KWARGS: Mapping[str, Any] = {
    "embedding_dim": 48,
    "attention_dim": 24,
    "temporal_hidden_dim": 48,
    "fusion_hidden_dim": 96,
    "dropout": 0.35,
    "modality_dropout": 0.20,
}
DEFAULT_TRAINING_SEED = 20_260_917


class SemanticFeatureLoadError(RuntimeError):
    """Raised when a requested clip cannot provide its own semantic features.

    Replacing a failed sample with the following dataset item silently changes
    the official split, duplicates labels, and can make an experiment look
    healthier than it is.  Training callers should precompute/audit the cache
    and fix or explicitly exclude failed source clips instead.
    """


def seed_everything(seed: int | None, *, deterministic: bool = True) -> int | None:
    """Seed Python, NumPy, Torch, and CUDA for a repeatable experiment.

    ``None`` intentionally leaves random state alone.  Deterministic Torch
    mode is warn-only because a platform-specific unsupported operation should
    be visible without making a long-running CPU cache experiment impossible.
    """

    if seed is None:
        return None
    resolved = int(seed)
    if not 0 <= resolved < 2**63:
        raise ValueError("seed must be in [0, 2**63).")
    random.seed(resolved)
    np.random.seed(resolved % (2**32))
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    return resolved


def _seed_data_loader_worker(_worker_id: int) -> None:
    """Seed third-party RNGs in spawned DataLoader workers."""

    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def compact_semantic_five_pillar_config(
    input_dims: Mapping[str, int],
    **overrides: Any,
) -> SemanticFivePillarConfig:
    """Build the small default architecture used by the LIRIS trainer.

    Keeping this factory in the training module avoids changing the generic
    model's public defaults while making the experiment choice explicit in a
    checkpoint and easy to override from the CLI/API.
    """

    values = dict(DEFAULT_COMPACT_MODEL_KWARGS)
    values.update(overrides)
    return SemanticFivePillarConfig(input_dims=dict(input_dims), **values)


@dataclass(frozen=True)
class LirisRecord:
    """One raw LIRIS clip joined to its official target and split."""

    video_id: str
    video_path: str
    valence_rank: int
    label: int
    split: str


def rank_to_class(valence_rank: int) -> int:
    """Map an official LIRIS valence rank to Positive/Neutral/Negative.

    The mapping is deliberately based on the whole 9,800-clip rank range,
    before inspecting any split.  It prevents the prior score-threshold
    procedure from creating a dominant Neutral class.
    """

    rank = int(valence_rank)
    if not 0 <= rank < TOTAL_LIRIS_RANKS:
        raise ValueError(
            f"valence_rank must be in [0, {TOTAL_LIRIS_RANKS - 1}], got {rank}."
        )
    lower_third = TOTAL_LIRIS_RANKS // 3  # 3266
    upper_third = 2 * TOTAL_LIRIS_RANKS // 3  # 6533
    if rank < lower_third:
        return 2  # Negative
    if rank < upper_third:
        return 1  # Neutral
    return 0  # Positive


def _read_tsv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Annotation file was not found: {target}")
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Annotation file has no TSV header: {target}")
        return [dict(row) for row in reader]


def _required_column(rows: Sequence[Mapping[str, str]], column: str, path: Path) -> None:
    if not rows or column not in rows[0]:
        available = sorted(rows[0]) if rows else []
        raise ValueError(
            f"Expected TSV column {column!r} in {path}; found {available}."
        )


def _video_index(video_dir: str | Path) -> dict[str, Path]:
    root = Path(video_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"LIRIS video directory was not found: {root}")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: (path.name.casefold(), str(path)),
    )
    if not paths:
        raise ValueError(f"No .mp4 videos were found under: {root}")
    index: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in paths:
        key = path.name
        if key in index:
            duplicates.append(key)
        else:
            index[key] = path
    if duplicates:
        examples = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            "Video basenames must be unique to join LIRIS annotations; "
            f"duplicates include: {examples}."
        )
    return index


def build_liris_records(
    video_dir: str | Path,
    ranking_path: str | Path,
    sets_path: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, list[LirisRecord]]:
    """Join LIRIS raw videos to ranking labels and official split membership.

    ``ACCEDEsets.txt`` uses ``1`` for train, ``2`` for validation, and ``0``
    for test.  By default the function refuses a partial join so a typo in a
    dataset path cannot silently alter class balance or test protocol.
    """

    ranking_file = Path(ranking_path)
    sets_file = Path(sets_path)
    ranking_rows = _read_tsv(ranking_file)
    set_rows = _read_tsv(sets_file)
    _required_column(ranking_rows, "name", ranking_file)
    _required_column(ranking_rows, "valenceRank", ranking_file)
    _required_column(set_rows, "name", sets_file)
    _required_column(set_rows, "set", sets_file)

    ranks: dict[str, int] = {}
    for row in ranking_rows:
        name = str(row.get("name", "")).strip()
        rank_value = str(row.get("valenceRank", "")).strip()
        if not name or not rank_value:
            raise ValueError(f"Incomplete ranking row in {ranking_file}: {row}")
        if name in ranks:
            raise ValueError(f"Duplicate ranking video name {name!r} in {ranking_file}")
        try:
            ranks[name] = int(float(rank_value))
        except ValueError as error:
            raise ValueError(f"Invalid valenceRank for {name!r}: {rank_value!r}") from error
        rank_to_class(ranks[name])  # validate the official range early

    split_codes: dict[str, str] = {}
    for row in set_rows:
        name = str(row.get("name", "")).strip()
        code = str(row.get("set", "")).strip()
        if not name or not code:
            raise ValueError(f"Incomplete split row in {sets_file}: {row}")
        if code not in OFFICIAL_SPLIT_CODES:
            raise ValueError(f"Unknown official split code {code!r} for {name!r}")
        if name in split_codes:
            raise ValueError(f"Duplicate split video name {name!r} in {sets_file}")
        split_codes[name] = code

    rank_only = sorted(set(ranks).difference(split_codes))
    split_only = sorted(set(split_codes).difference(ranks))
    if rank_only or split_only:
        detail: list[str] = []
        if rank_only:
            detail.append(f"ranking-only={rank_only[:3]}")
        if split_only:
            detail.append(f"split-only={split_only[:3]}")
        raise ValueError("Ranking and split annotation files do not join exactly: " + "; ".join(detail))

    videos = _video_index(video_dir)
    missing_videos = sorted(set(ranks).difference(videos))
    if missing_videos and require_complete:
        raise FileNotFoundError(
            f"{len(missing_videos)} annotated LIRIS clips are missing below {Path(video_dir)}; "
            f"examples: {missing_videos[:5]}. Set require_complete=False only for a deliberate smoke test."
        )

    records: dict[str, list[LirisRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for name, rank in ranks.items():
        path = videos.get(name)
        if path is None:
            continue
        split = OFFICIAL_SPLIT_CODES[split_codes[name]]
        records[split].append(
            LirisRecord(
                video_id=path.stem,
                video_path=str(path),
                valence_rank=rank,
                label=rank_to_class(rank),
                split=split,
            )
        )
    for split in records:
        records[split].sort(key=lambda record: record.video_id)

    if not all(records.values()):
        counts = {split: len(rows) for split, rows in records.items()}
        raise ValueError(f"No complete official split was built. Counts: {counts}")
    return records


def records_summary(records: Mapping[str, Sequence[LirisRecord]]) -> dict[str, Any]:
    """Return JSON-safe split and class counts for experiment logging."""

    result: dict[str, Any] = {}
    for split, rows in records.items():
        class_counts = {CLASS_NAMES[index]: 0 for index in range(len(CLASS_NAMES))}
        for record in rows:
            class_counts[CLASS_NAMES[record.label]] += 1
        result[str(split)] = {"clips": len(rows), "class_counts": class_counts}
    return result


def _source_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _safe_cache_stem(video_id: str) -> str:
    """Return a filename-safe, deterministic stem without path traversal."""

    sanitized = "".join(character if character.isalnum() or character in "-_." else "_" for character in video_id)
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError(f"Unsafe video id for cache filename: {video_id!r}")
    return sanitized


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy/torch values to checkpoint metadata values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    """Write small metadata atomically, avoiding partially-written caches."""

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
    """Write an ``npz`` payload atomically in the cache directory."""

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
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _audio_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract durable audio status/error fields from semantic evidence."""

    evidence = result.get("evidence", {})
    audio = evidence.get("audio", {}) if isinstance(evidence, Mapping) else {}
    if not isinstance(audio, Mapping):
        audio = {}
    # Retain the full small audio evidence mapping as well as top-level fields
    # that are easy to aggregate later.  This is important: decode failures
    # must not be mistaken for genuinely silent clips.
    return {
        "audio_status": str(audio.get("audio_status", "unknown")),
        "audio_error_type": (
            None if audio.get("audio_error_type") is None else str(audio.get("audio_error_type"))
        ),
        "audio_note": None if audio.get("note") is None else str(audio.get("note")),
        "audio_evidence": _json_safe(dict(audio)),
    }


class LirisSemanticFivePillarDataset(Dataset[dict[str, Any]]):
    """Official LIRIS records backed by versioned semantic-feature caches.

    Cache files are a pair of ``.npz`` arrays and ``.json`` metadata.  The
    pair is accepted only if its extractor signature and source file size/time
    fingerprint match.  If either half is missing or stale, the video is
    re-extracted.  Extraction exceptions deliberately propagate to callers;
    use :meth:`precompute` to collect a failure report before a long run.
    """

    def __init__(
        self,
        records: Sequence[LirisRecord],
        cache_dir: str | Path,
        *,
        extractor_config: SemanticPillarConfig | None = None,
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("A semantic LIRIS dataset needs at least one record.")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = SemanticPillarExtractor(extractor_config)
        self.extractor_signature = self.extractor.cache_signature()

    def __len__(self) -> int:
        return len(self.records)

    def _cache_paths(self, record: LirisRecord) -> tuple[Path, Path]:
        stem = _safe_cache_stem(record.video_id)
        return self.cache_dir / f"{stem}.npz", self.cache_dir / f"{stem}.json"

    def _cache_is_current(self, metadata: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
        return (
            metadata.get("cache_format") == CACHE_FORMAT_VERSION
            and metadata.get("extractor_schema_version") == SCHEMA_VERSION
            and metadata.get("extractor_signature") == self.extractor_signature
            and metadata.get("source") == dict(source)
        )

    def _validate_cached_arrays(self, payload: Mapping[str, np.ndarray]) -> None:
        lengths: set[int] = set()
        expected_dims = self.extractor.pillar_dims
        for name in PILLAR_NAMES:
            values = payload[f"feature_{name}"]
            available = payload[f"availability_{name}"]
            reliable = payload[f"reliability_{name}"]
            if values.ndim != 2 or values.shape[1] != expected_dims[name]:
                raise ValueError(f"Cached {name} features have invalid shape {values.shape}.")
            if available.ndim != 1 or reliable.ndim != 1:
                raise ValueError(f"Cached {name} masks must be one dimensional.")
            if values.shape[0] != available.shape[0] or values.shape[0] != reliable.shape[0]:
                raise ValueError(f"Cached {name} arrays have inconsistent sequence lengths.")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Cached {name} features contain non-finite values.")
            if not np.all(np.isfinite(available)) or not np.all(np.isfinite(reliable)):
                raise ValueError(f"Cached {name} masks contain non-finite values.")
            if np.any(available < 0.0) or np.any(available > 1.0):
                raise ValueError(f"Cached {name} availability is outside [0, 1].")
            if np.any(reliable < 0.0) or np.any(reliable > 1.0):
                raise ValueError(f"Cached {name} reliability is outside [0, 1].")
            lengths.add(int(values.shape[0]))
        if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
            raise ValueError("Cached pillar arrays must share one non-empty sequence length.")

    def _read_cache(
        self,
        record: LirisRecord,
        source: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]] | None:
        npz_path, metadata_path = self._cache_paths(record)
        if not npz_path.is_file() or not metadata_path.is_file():
            return None
        metadata = _read_json(metadata_path)
        if metadata is None or not self._cache_is_current(metadata, source):
            return None
        try:
            with np.load(npz_path, allow_pickle=False) as loaded:
                arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
            required = {
                *[f"feature_{name}" for name in PILLAR_NAMES],
                *[f"availability_{name}" for name in PILLAR_NAMES],
                *[f"reliability_{name}" for name in PILLAR_NAMES],
            }
            if set(arrays) != required:
                return None
            self._validate_cached_arrays(arrays)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):  # type: ignore[name-defined]
            return None
        features = {
            name: np.asarray(arrays[f"feature_{name}"], dtype=np.float32)
            for name in PILLAR_NAMES
        }
        availability = {
            name: np.asarray(arrays[f"availability_{name}"], dtype=np.float32)
            for name in PILLAR_NAMES
        }
        reliability = {
            name: np.asarray(arrays[f"reliability_{name}"], dtype=np.float32)
            for name in PILLAR_NAMES
        }
        return features, availability, reliability, metadata

    def _write_cache(
        self,
        record: LirisRecord,
        source: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
        features_raw = result.get("features")
        availability_raw = result.get("availability")
        reliability_raw = result.get("reliability")
        if not isinstance(features_raw, Mapping) or not isinstance(availability_raw, Mapping) or not isinstance(reliability_raw, Mapping):
            raise RuntimeError("Semantic extractor returned an incomplete feature schema.")
        arrays: dict[str, np.ndarray] = {}
        for name in PILLAR_NAMES:
            try:
                arrays[f"feature_{name}"] = np.asarray(features_raw[name], dtype=np.float32)
                arrays[f"availability_{name}"] = np.asarray(availability_raw[name], dtype=np.float32)
                arrays[f"reliability_{name}"] = np.asarray(reliability_raw[name], dtype=np.float32)
            except KeyError as error:
                raise RuntimeError(f"Semantic extractor omitted pillar {name!r}.") from error
        self._validate_cached_arrays(arrays)

        npz_path, metadata_path = self._cache_paths(record)
        metadata: dict[str, Any] = {
            "cache_format": CACHE_FORMAT_VERSION,
            "extractor_schema_version": str(result.get("schema_version", "")),
            "extractor_signature": self.extractor_signature,
            "extractor_config": self.extractor.config_metadata(),
            "source": dict(source),
            "video_id": record.video_id,
            "sequence_length": int(arrays[f"feature_{PILLAR_NAMES[0]}"] .shape[0]),
            "audio": _audio_metadata(result),
            "video": _json_safe(result.get("video", {})),
        }
        # The arrays are committed before metadata.  A crash between the two
        # leaves an orphan array that is rejected and safely replaced later.
        _atomic_npz_dump(npz_path, arrays)
        _atomic_json_dump(metadata_path, metadata)
        features = {name: arrays[f"feature_{name}"] for name in PILLAR_NAMES}
        availability = {name: arrays[f"availability_{name}"] for name in PILLAR_NAMES}
        reliability = {name: arrays[f"reliability_{name}"] for name in PILLAR_NAMES}
        return features, availability, reliability, metadata

    def load_or_extract(
        self, record: LirisRecord
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
        """Return validated cache data or extract it and write a fresh cache."""

        source = _source_fingerprint(record.video_path)
        cached = self._read_cache(record, source)
        if cached is not None:
            return cached
        # LIRIS nominally contains short clips, but two source files exceed
        # the public 15-second upload limit.  Cache their first 15 seconds so
        # the official split remains complete; upload inference still rejects
        # user videos over the limit.
        result = self.extractor.extract(record.video_path, truncate_to_limit=True)
        return self._write_cache(record, source, result)

    def audit_cache(self, *, example_limit: int = 10) -> dict[str, Any]:
        """Inspect cache validity and audio coverage without writing anything.

        This deliberately uses the same validation path as normal loading but
        never falls back to extraction.  It gives an experiment runner a
        concise, serialisable answer to three questions before training: are
        all source fingerprints current, which records need attention, and is
        one split unexpectedly missing a modality?
        """

        if example_limit < 0:
            raise ValueError("example_limit must be zero or positive.")
        current = 0
        missing_or_stale = 0
        source_errors = 0
        audio_status_counts: dict[str, int] = {}
        audio_usable_records = 0
        examples: list[dict[str, str]] = []

        for record in self.records:
            try:
                source = _source_fingerprint(record.video_path)
            except OSError as error:
                source_errors += 1
                if len(examples) < example_limit:
                    examples.append(
                        {
                            "video_id": record.video_id,
                            "reason": f"source_error:{type(error).__name__}",
                        }
                    )
                continue
            cached = self._read_cache(record, source)
            if cached is None:
                missing_or_stale += 1
                if len(examples) < example_limit:
                    examples.append({"video_id": record.video_id, "reason": "missing_or_stale_cache"})
                continue

            _features, availability, reliability, metadata = cached
            current += 1
            audio = metadata.get("audio", {}) if isinstance(metadata, Mapping) else {}
            status = "unknown"
            if isinstance(audio, Mapping):
                status = str(audio.get("audio_status", status))
            audio_status_counts[status] = audio_status_counts.get(status, 0) + 1
            if np.any((availability["audio"] > 0.0) & (reliability["audio"] > 0.0)):
                audio_usable_records += 1

        total = len(self.records)
        return {
            "records": total,
            "current": current,
            "missing_or_stale": missing_or_stale,
            "source_errors": source_errors,
            "all_current": current == total,
            "audio_status_counts": dict(sorted(audio_status_counts.items())),
            "audio_usable_records": audio_usable_records,
            "audio_usable_fraction": float(audio_usable_records / total) if total else 0.0,
            "examples": examples,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            features, availability, reliability, metadata = self.load_or_extract(record)
        except Exception as exc:
            raise SemanticFeatureLoadError(
                f"Could not load semantic features for {record.video_id} at {record.video_path}. "
                "Run precompute/audit, repair the cache or source clip, then retry; "
                "the dataset will not substitute a different labeled clip."
            ) from exc

        audio = metadata.get("audio", {}) if isinstance(metadata, Mapping) else {}
        return {
            "sequences": {name: torch.from_numpy(features[name].copy()) for name in PILLAR_NAMES},
            "availability": {name: torch.from_numpy(availability[name].copy()) for name in PILLAR_NAMES},
            "reliability": {name: torch.from_numpy(reliability[name].copy()) for name in PILLAR_NAMES},
            "label": torch.tensor(record.label, dtype=torch.long),
            "video_id": record.video_id,
            "audio_status": str(audio.get("audio_status", "unknown")),
            "audio_error_type": audio.get("audio_error_type"),
        }

    def precompute(
        self,
        limit: int | None = None,
        *,
        workers: int = 0,
        worker_threads: int | None = None,
        progress_every: int = 25,
    ) -> dict[str, Any]:
        """Populate this cache and return visible success/failure diagnostics.

        Failures are collected instead of being presented as successful mute
        clips.  They are not stored as feature caches, so a repaired decoder or
        file can be retried on the next run.

        ``workers=0`` retains the original in-process behavior.  A positive
        value creates Windows ``spawn`` workers, each with one long-lived
        extractor.  That is important for the optional CPU ResNet: creating an
        extractor per clip would repeatedly load its weights and be slower than
        the serial path.  Every worker calls :meth:`load_or_extract`, so cache
        signatures, source fingerprints, metadata, and atomic cache commits
        are exactly the same as the serial path.

        Parallel workers never download model weights.  If pretrained spatial
        context is wanted, its weights must already be present locally and the
        supplied configuration must keep ``allow_model_download=False``.
        """

        count = len(self.records) if limit is None else min(max(int(limit), 0), len(self.records))
        selected = self.records[:count]
        if workers < 0:
            raise ValueError("workers must be zero (serial) or a positive integer.")
        if worker_threads is not None and int(worker_threads) < 1:
            raise ValueError("worker_threads must be positive when supplied.")
        if progress_every < 0:
            raise ValueError("progress_every must be zero or positive.")

        if workers == 0 or count == 0:
            return _precompute_records_serial(
                self,
                selected,
                progress_every=progress_every,
            )
        if self.extractor.config.allow_model_download:
            raise ValueError(
                "Parallel cache precomputation never downloads pretrained weights. "
                "Cache the weights first, then rerun with allow_model_download=False."
            )
        return _precompute_records_parallel(
            selected,
            cache_dir=self.cache_dir,
            extractor_config=self.extractor.config,
            workers=workers,
            worker_threads=worker_threads,
            progress_every=progress_every,
        )


# ``spawn`` workers re-import this module on Windows.  The worker state is
# deliberately module-global rather than a bound Dataset method: top-level
# functions and these small dataclasses are pickle-safe, while the extractor
# itself remains process-local and is constructed exactly once per worker.
_PRECOMPUTE_WORKER_DATASET: Any | None = None


def _precompute_success_event(record: LirisRecord, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one successful cache read/write into the serial report schema."""

    audio = metadata.get("audio", {}) if isinstance(metadata, Mapping) else {}
    if not isinstance(audio, Mapping):
        audio = {}
    error_type = audio.get("audio_error_type")
    return {
        "ok": True,
        "video_id": record.video_id,
        "audio_status": str(audio.get("audio_status", "unknown")),
        "audio_error_type": None if error_type is None else str(error_type),
    }


def _precompute_failure_event(record: LirisRecord, error: Exception) -> dict[str, Any]:
    """Keep decode/extraction exceptions visible instead of treating them as mute."""

    return {
        "ok": False,
        "video_id": record.video_id,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _new_precompute_report(requested: int) -> dict[str, Any]:
    return {
        "requested": int(requested),
        "completed": 0,
        "failed": 0,
        "failures": [],
        "audio_status_counts": {},
        "audio_error_type_counts": {},
    }


def _add_precompute_event(report: dict[str, Any], event: Mapping[str, Any]) -> None:
    """Aggregate an event produced by either the serial or spawned path."""

    if bool(event.get("ok")):
        report["completed"] = int(report["completed"]) + 1
        status = str(event.get("audio_status", "unknown"))
        status_counts = report["audio_status_counts"]
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        error_type = event.get("audio_error_type")
        if error_type:
            value = str(error_type)
            error_counts = report["audio_error_type_counts"]
            error_counts[value] = int(error_counts.get(value, 0)) + 1
        return
    report["failed"] = int(report["failed"]) + 1
    report["failures"].append(
        {
            "video_id": str(event.get("video_id", "unknown")),
            "error_type": str(event.get("error_type", "RuntimeError")),
            "message": str(event.get("message", "Unknown precompute failure.")),
        }
    )


def _format_precompute_duration(seconds: float) -> str:
    """Compact elapsed/ETA text for a long local cache run."""

    whole_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds_part:02d}s"
    return f"{seconds_part}s"


def _emit_precompute_progress(
    processed: int,
    total: int,
    *,
    started_at: float,
    workers: int,
    progress_every: int,
) -> None:
    """Print parent-only progress; child workers never contend for stdout."""

    if progress_every == 0 or (processed % progress_every != 0 and processed != total):
        return
    elapsed = max(0.0, time.monotonic() - started_at)
    if processed and elapsed > 0.0:
        rate_per_minute = processed / elapsed * 60.0
        remaining = max(0, total - processed)
        eta = _format_precompute_duration(remaining / (processed / elapsed))
        speed = f"{rate_per_minute:.1f} clips/min, ETA {eta}"
    else:
        speed = "estimating speed"
    mode = "serial" if workers == 0 else f"{workers} worker(s)"
    print(
        f"[semantic-cache] {processed}/{total} clips processed ({mode}; {speed})",
        flush=True,
    )


def _precompute_records_serial(
    dataset: LirisSemanticFivePillarDataset,
    records: Sequence[LirisRecord],
    *,
    progress_every: int,
) -> dict[str, Any]:
    """Original behavior, refactored to share report logic with parallel work."""

    report = _new_precompute_report(len(records))
    started_at = time.monotonic()
    for processed, record in enumerate(records, start=1):
        try:
            _, _, _, metadata = dataset.load_or_extract(record)
            event = _precompute_success_event(record, metadata)
        except Exception as error:  # report, do not suppress training-time errors
            event = _precompute_failure_event(record, error)
        _add_precompute_event(report, event)
        _emit_precompute_progress(
            processed,
            len(records),
            started_at=started_at,
            workers=0,
            progress_every=progress_every,
        )
    return report


def _resolve_precompute_worker_threads(workers: int, requested: int | None) -> int:
    """Choose a conservative per-process thread cap for CPU video features."""

    if requested is not None:
        return int(requested)
    logical_cpus = os.cpu_count() or 1
    # The extractors use both PyTorch (ResNet) and OpenCV (optical flow).  A
    # small cap prevents N spawned processes from each trying to use every core.
    return max(1, min(4, logical_cpus // max(1, workers)))


def _configure_precompute_worker_threads(worker_threads: int) -> None:
    """Limit libraries inside a fresh spawn worker to avoid CPU oversubscription."""

    try:
        torch.set_num_threads(worker_threads)
    except (RuntimeError, TypeError):
        pass
    try:
        torch.set_num_interop_threads(1)
    except (RuntimeError, TypeError):
        # PyTorch only permits this once per process.  A fresh spawn worker
        # normally reaches this call first, but preserving progress is safer
        # than failing a cache run if another library initialized it earlier.
        pass
    try:
        import cv2

        cv2.setNumThreads(worker_threads)
    except (ImportError, AttributeError, TypeError):
        # The extractor will later report a genuine OpenCV dependency problem
        # per clip.  Do not mask it during worker configuration.
        pass


def _init_precompute_worker(
    cache_dir: str,
    extractor_config: SemanticPillarConfig,
    seed_record: LirisRecord,
    worker_threads: int,
) -> None:
    """Create one reusable extractor/cache adapter in each spawned process."""

    global _PRECOMPUTE_WORKER_DATASET
    if extractor_config.allow_model_download:
        raise RuntimeError("Parallel semantic cache workers never download pretrained weights.")
    _configure_precompute_worker_threads(worker_threads)
    _PRECOMPUTE_WORKER_DATASET = LirisSemanticFivePillarDataset(
        [seed_record],
        cache_dir,
        extractor_config=extractor_config,
    )


def _precompute_record_worker(record: LirisRecord) -> dict[str, Any]:
    """Top-level, picklable worker task used by Windows ``spawn`` processes."""

    dataset = _PRECOMPUTE_WORKER_DATASET
    if not isinstance(dataset, LirisSemanticFivePillarDataset):
        raise RuntimeError("Semantic cache worker was not initialized.")
    try:
        _, _, _, metadata = dataset.load_or_extract(record)
        return _precompute_success_event(record, metadata)
    except Exception as error:
        return _precompute_failure_event(record, error)


def _precompute_records_parallel(
    records: Sequence[LirisRecord],
    *,
    cache_dir: Path,
    extractor_config: SemanticPillarConfig,
    workers: int,
    worker_threads: int | None,
    progress_every: int,
) -> dict[str, Any]:
    """Fill cache entries concurrently using Windows-safe spawned processes.

    No cache state is passed between processes.  Each task invokes the same
    Dataset cache reader/writer as serial extraction, whose ``os.replace``
    commits keep every array/metadata pair valid if a process is interrupted.
    """

    if workers < 1:
        raise ValueError("Parallel precomputation requires at least one worker.")
    if extractor_config.allow_model_download:
        raise ValueError("Parallel semantic cache workers never download pretrained weights.")
    if str(extractor_config.spatial_device).strip().lower() not in {"cpu", "cpu:0"}:
        raise ValueError(
            "Parallel semantic cache extraction requires spatial_device='cpu'. "
            "Use serial precomputation for a GPU spatial encoder."
        )

    cache_stems = [_safe_cache_stem(record.video_id).casefold() for record in records]
    if len(cache_stems) != len(set(cache_stems)):
        raise ValueError("Parallel precomputation needs unique video IDs/cache filenames.")

    resolved_threads = _resolve_precompute_worker_threads(workers, worker_threads)
    report = _new_precompute_report(len(records))
    started_at = time.monotonic()
    # Explicit spawn avoids inheriting an OpenCV/Torch process state on Unix
    # and is the only portable choice on Windows.  The main entry point remains
    # guarded below, so child imports do not start another CLI invocation.
    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=spawn_context,
        initializer=_init_precompute_worker,
        initargs=(str(cache_dir), extractor_config, records[0], resolved_threads),
    ) as executor:
        future_to_record: dict[Future[dict[str, Any]], LirisRecord] = {
            executor.submit(_precompute_record_worker, record): record for record in records
        }
        for processed, future in enumerate(as_completed(future_to_record), start=1):
            record = future_to_record[future]
            try:
                event = future.result()
            except Exception as error:
                # An unexpected worker-side failure remains tied to the exact
                # record.  Other tasks still drain and report their outcomes.
                event = _precompute_failure_event(record, error)
            _add_precompute_event(report, event)
            _emit_precompute_progress(
                processed,
                len(records),
                started_at=started_at,
                workers=workers,
                progress_every=progress_every,
            )
    return report


def semantic_five_pillar_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad variable-length semantic sequences while retaining zero masks.

    This function pads only at the end, which is the prefix-mask invariant
    required by the fusion model's packed temporal encoder.  Audio remains
    zero-masked for mute videos, rather than receiving a fabricated mean.
    """

    if not batch:
        raise ValueError("Cannot collate an empty semantic five-pillar batch.")
    lengths = [int(item["sequences"][PILLAR_NAMES[0]].shape[0]) for item in batch]
    if any(length < 1 for length in lengths):
        raise ValueError("Semantic sequences must have at least one window.")
    batch_size = len(batch)
    max_steps = max(lengths)
    sequences: dict[str, Tensor] = {}
    availability: dict[str, Tensor] = {}
    reliability: dict[str, Tensor] = {}
    for name in PILLAR_NAMES:
        first_values = torch.as_tensor(batch[0]["sequences"][name], dtype=torch.float32)
        if first_values.ndim != 2:
            raise ValueError(
                f"Batch item 0 has invalid {name} shape {tuple(first_values.shape)}; expected (T, D)."
            )
        dim = int(first_values.shape[1])
        if dim < 1:
            raise ValueError(f"Batch item 0 has no {name} feature dimensions.")
        padded_values = torch.zeros(batch_size, max_steps, dim, dtype=torch.float32)
        padded_availability = torch.zeros(batch_size, max_steps, dtype=torch.float32)
        padded_reliability = torch.zeros(batch_size, max_steps, dtype=torch.float32)
        for row, item in enumerate(batch):
            values = torch.as_tensor(item["sequences"][name], dtype=torch.float32)
            available = torch.as_tensor(item["availability"][name], dtype=torch.float32)
            reliable = torch.as_tensor(item["reliability"][name], dtype=torch.float32)
            length = lengths[row]
            if tuple(values.shape) != (length, dim):
                raise ValueError(
                    f"Batch item {row} has invalid {name} shape {tuple(values.shape)}; expected {(length, dim)}."
                )
            if tuple(available.shape) != (length,) or tuple(reliable.shape) != (length,):
                raise ValueError(f"Batch item {row} has invalid {name} mask shapes.")
            if not torch.isfinite(values).all() or not torch.isfinite(available).all() or not torch.isfinite(reliable).all():
                raise ValueError(f"Batch item {row} has non-finite {name} data.")
            padded_values[row, :length] = values
            padded_availability[row, :length] = available.clamp(0.0, 1.0)
            padded_reliability[row, :length] = reliable.clamp(0.0, 1.0)
        sequences[name] = padded_values
        availability[name] = padded_availability
        reliability[name] = padded_reliability
    return {
        "sequences": sequences,
        "availability": availability,
        "reliability": reliability,
        "labels": torch.as_tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "lengths": torch.as_tensor(lengths, dtype=torch.long),
        "video_ids": [str(item["video_id"]) for item in batch],
        "audio_statuses": [str(item.get("audio_status", "unknown")) for item in batch],
        "audio_error_types": [item.get("audio_error_type") for item in batch],
    }


@dataclass(frozen=True)
class PillarFeatureNormalizer:
    """Per-pillar z-score state fit only on valid train windows.

    Values with zero availability or zero reliability are kept at zero when
    transformed.  That is crucial for mute audio: standardisation must not
    turn an absent zero-vector into an artificial mean feature vector.
    """

    statistics: Mapping[str, Mapping[str, Sequence[float]]]
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        keys = set(self.statistics)
        if keys != set(PILLAR_NAMES):
            raise ValueError(f"Normalizer pillars must be exactly {PILLAR_NAMES}; got {sorted(keys)}.")
        for name in PILLAR_NAMES:
            values = self.statistics[name]
            mean = np.asarray(values.get("mean", []), dtype=np.float32)
            std = np.asarray(values.get("std", []), dtype=np.float32)
            if mean.ndim != 1 or mean.size < 1 or std.shape != mean.shape:
                raise ValueError(f"Invalid normalizer shape for {name}: {mean.shape}/{std.shape}.")
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
                raise ValueError(f"Invalid normalizer values for {name}.")

    @classmethod
    def fit(cls, dataset: LirisSemanticFivePillarDataset, *, epsilon: float = 1e-6) -> "PillarFeatureNormalizer":
        """Fit on the training dataset's nonzero-quality feature windows only."""

        dimensions = dict(dataset.extractor.pillar_dims)
        missing = set(PILLAR_NAMES).difference(dimensions)
        if missing:
            raise ValueError(f"Extractor dimensions are missing pillars: {sorted(missing)}")
        sums = {name: np.zeros(int(dimensions[name]), dtype=np.float64) for name in PILLAR_NAMES}
        sum_squares = {name: np.zeros(int(dimensions[name]), dtype=np.float64) for name in PILLAR_NAMES}
        counts = {name: 0 for name in PILLAR_NAMES}
        for record in dataset.records:
            try:
                features, availability, reliability, _ = dataset.load_or_extract(record)
            except Exception as exc:
                raise SemanticFeatureLoadError(
                    f"Could not fit normalizer because {record.video_id} has no usable semantic "
                    "features. Precompute/audit the exact clip instead of silently changing the "
                    "training distribution."
                ) from exc
            for name in PILLAR_NAMES:
                valid = (availability[name] > 0.0) & (reliability[name] > 0.0)
                if not np.any(valid):
                    continue
                values = np.asarray(features[name][valid], dtype=np.float64)
                if values.ndim != 2 or values.shape[1] != int(dimensions[name]):
                    raise SemanticFeatureLoadError(
                        f"{record.video_id} has {name} features shaped {values.shape}; expected "
                        f"(*, {int(dimensions[name])})."
                    )
                sums[name] += values.sum(axis=0)
                sum_squares[name] += np.square(values).sum(axis=0)
                counts[name] += int(values.shape[0])
        statistics: dict[str, dict[str, list[float]]] = {}
        for name in PILLAR_NAMES:
            if counts[name] == 0:
                # A fully unavailable training modality can still be supplied
                # at inference; identity normalisation is the only honest
                # fallback.  Its zero masks keep it excluded from fusion.
                mean = np.zeros(int(dimensions[name]), dtype=np.float64)
                std = np.ones(int(dimensions[name]), dtype=np.float64)
            else:
                mean = sums[name] / counts[name]
                variance = sum_squares[name] / counts[name] - np.square(mean)
                std = np.sqrt(np.maximum(variance, epsilon**2))
            statistics[name] = {
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
            }
        return cls(statistics=statistics, epsilon=epsilon)

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-safe checkpoint payload."""

        return {
            "format": "pillar-feature-normalizer-v2",
            "epsilon": float(self.epsilon),
            "input_dims": {
                name: int(len(self.statistics[name]["mean"])) for name in PILLAR_NAMES
            },
            "pillars": {
                name: {
                    "mean": list(self.statistics[name]["mean"]),
                    "std": list(self.statistics[name]["std"]),
                }
                for name in PILLAR_NAMES
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PillarFeatureNormalizer":
        """Restore either the canonical checkpoint state or bare pillars."""

        if "pillars" in state:
            values = state["pillars"]
            epsilon = float(state.get("epsilon", 1e-6))
        else:
            values = {name: state[name] for name in PILLAR_NAMES if name in state}
            epsilon = 1e-6
        if not isinstance(values, Mapping):
            raise ValueError("Normalizer checkpoint has no pillar mapping.")
        return cls(statistics=values, epsilon=epsilon)

    def apply(
        self,
        sequences: Mapping[str, Tensor],
        availability: Mapping[str, Tensor],
        reliability: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Z-score valid windows and force unavailable/padded windows to zero."""

        normalized: dict[str, Tensor] = {}
        for name in PILLAR_NAMES:
            values = sequences[name]
            mean = torch.as_tensor(self.statistics[name]["mean"], dtype=values.dtype, device=values.device)
            std = torch.as_tensor(self.statistics[name]["std"], dtype=values.dtype, device=values.device)
            if values.ndim != 3 or values.shape[-1] != mean.numel():
                raise ValueError(
                    f"{name} sequences must have shape (B, T, {mean.numel()}); got {tuple(values.shape)}."
                )
            valid = (availability[name] > 0.0) & (reliability[name] > 0.0)
            scaled = (values - mean.view(1, 1, -1)) / std.view(1, 1, -1).clamp_min(self.epsilon)
            normalized[name] = torch.where(valid.unsqueeze(-1), scaled, torch.zeros_like(scaled))
        return normalized


def compute_class_weights(records: Sequence[LirisRecord], device: torch.device | str = "cpu") -> Tensor:
    """Return inverse-frequency class weights from *training records only*."""

    counts = np.bincount([record.label for record in records], minlength=len(CLASS_NAMES)).astype(np.float32)
    if np.any(counts <= 0):
        raise ValueError(f"Every class must occur in training records; counts={counts.tolist()}.")
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=torch.device(device))


def build_liris_datasets(
    video_dir: str | Path,
    ranking_path: str | Path,
    sets_path: str | Path,
    cache_dir: str | Path,
    *,
    extractor_config: SemanticPillarConfig | None = None,
    require_complete: bool = True,
) -> tuple[dict[str, LirisSemanticFivePillarDataset], dict[str, list[LirisRecord]]]:
    """Build datasets using only official LIRIS labels and raw videos."""

    records = build_liris_records(
        video_dir,
        ranking_path,
        sets_path,
        require_complete=require_complete,
    )
    cache_root = Path(cache_dir)
    datasets = {
        split: LirisSemanticFivePillarDataset(
            split_records,
            cache_root / split,
            extractor_config=extractor_config,
        )
        for split, split_records in records.items()
    }
    return datasets, records


def build_liris_loaders(
    video_dir: str | Path,
    ranking_path: str | Path,
    sets_path: str | Path,
    cache_dir: str | Path,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int | None = None,
    extractor_config: SemanticPillarConfig | None = None,
    require_complete: bool = True,
) -> tuple[dict[str, DataLoader], dict[str, LirisSemanticFivePillarDataset], dict[str, list[LirisRecord]]]:
    """Build official-split DataLoaders and expose their source datasets."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    datasets, records = build_liris_datasets(
        video_dir,
        ranking_path,
        sets_path,
        cache_dir,
        extractor_config=extractor_config,
        require_complete=require_complete,
    )
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": semantic_five_pillar_collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        common["persistent_workers"] = True

    def loader_generator(offset: int) -> torch.Generator | None:
        if seed is None:
            return None
        generator = torch.Generator()
        generator.manual_seed((int(seed) + offset) % (2**63))
        return generator

    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=loader_generator(0),
            worker_init_fn=_seed_data_loader_worker if seed is not None else None,
            **common,
        ),
        "validation": DataLoader(
            datasets["validation"],
            shuffle=False,
            generator=loader_generator(1),
            worker_init_fn=_seed_data_loader_worker if seed is not None else None,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            generator=loader_generator(2),
            worker_init_fn=_seed_data_loader_worker if seed is not None else None,
            **common,
        ),
    }
    return loaders, datasets, records


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    """Compute accuracy, balanced accuracy, macro F1 and a confusion matrix."""

    if len(labels) != len(predictions) or not labels:
        if not labels and not predictions:
            return {
                "accuracy": float("nan"),
                "balanced_accuracy": float("nan"),
                "macro_f1": float("nan"),
                "confusion_matrix": [[0] * len(CLASS_NAMES) for _ in CLASS_NAMES],
                "per_class": {},
                "support": 0,
            }
        raise ValueError("Labels and predictions must be non-empty equal-length sequences.")
    number_classes = len(CLASS_NAMES)
    confusion = np.zeros((number_classes, number_classes), dtype=np.int64)
    for target, predicted in zip(labels, predictions, strict=True):
        if not 0 <= int(target) < number_classes or not 0 <= int(predicted) < number_classes:
            raise ValueError(f"Invalid class pair target={target}, prediction={predicted}.")
        confusion[int(target), int(predicted)] += 1
    support = confusion.sum(axis=1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted_count, out=np.zeros_like(true_positive), where=predicted_count > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    valid_recalls = recall[support > 0]
    per_class = {
        CLASS_NAMES[index]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index in range(number_classes)
    }
    return {
        "accuracy": float(true_positive.sum() / confusion.sum()),
        "balanced_accuracy": float(valid_recalls.mean()) if valid_recalls.size else float("nan"),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "support": int(confusion.sum()),
    }


def _model_config_payload(config: SemanticFivePillarConfig) -> dict[str, Any]:
    return {
        "input_dims": {name: int(config.input_dims[name]) for name in PILLAR_NAMES},
        "embedding_dim": int(config.embedding_dim),
        "attention_dim": int(config.attention_dim),
        "temporal_hidden_dim": int(config.temporal_hidden_dim),
        "fusion_hidden_dim": int(config.fusion_hidden_dim),
        "dropout": float(config.dropout),
        "modality_dropout": float(config.modality_dropout),
        "num_classes": int(config.num_classes),
        "class_names": list(config.class_names),
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a checkpoint so interruption never corrupts best.pt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp.pt"
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
        temporary = None  # type: ignore[assignment]
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


class SemanticFivePillarTrainer:
    """Train reliability-aware semantic fusion and preserve reproducibility data."""

    def __init__(
        self,
        model: SemanticFivePillarModel,
        *,
        normalizer: PillarFeatureNormalizer,
        extractor_config: SemanticPillarConfig,
        checkpoint_dir: str | Path,
        device: str | torch.device | None = None,
        class_weights: Tensor | None = None,
        auxiliary_weight: float = 0.10,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        if not 0.0 <= auxiliary_weight <= 1.0:
            raise ValueError("auxiliary_weight must be in [0, 1].")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.normalizer = normalizer
        self.extractor_config = extractor_config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.auxiliary_weight = float(auxiliary_weight)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.class_weights = (
            class_weights.detach().to(device=self.device, dtype=torch.float32).clone()
            if class_weights is not None
            else None
        )
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None
        self.best_validation_metrics: dict[str, Any] | None = None
        self.stopped_early = False
        self.stop_reason: str | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_semantic_five_pillar.pt"

    def _move_batch(
        self, batch: Mapping[str, Any]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], Tensor]:
        sequences = {name: batch["sequences"][name].to(self.device, non_blocking=True) for name in PILLAR_NAMES}
        availability = {name: batch["availability"][name].to(self.device, non_blocking=True) for name in PILLAR_NAMES}
        reliability = {name: batch["reliability"][name].to(self.device, non_blocking=True) for name in PILLAR_NAMES}
        labels = batch["labels"].to(self.device, non_blocking=True)
        normalized = self.normalizer.apply(sequences, availability, reliability)
        return normalized, availability, reliability, labels

    def _loss(self, output: Mapping[str, Any], labels: Tensor) -> tuple[Tensor, dict[str, float]]:
        main = self.criterion(output["logits"], labels)
        aux_losses: list[Tensor] = []
        fusion_mask = output["fusion_mask"]
        pillar_logits = output["pillar_logits"]
        for index, name in enumerate(PILLAR_NAMES):
            valid = fusion_mask[:, index].bool()
            if bool(valid.any()):
                aux_losses.append(self.criterion(pillar_logits[name][valid], labels[valid]))
        if aux_losses:
            auxiliary = torch.stack(aux_losses).mean()
            total = main + self.auxiliary_weight * auxiliary
        else:
            auxiliary = torch.zeros((), dtype=main.dtype, device=main.device)
            total = main
        return total, {"main_loss": float(main.detach().cpu()), "auxiliary_loss": float(auxiliary.detach().cpu())}

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        total_main_loss = 0.0
        total_auxiliary_loss = 0.0
        total_examples = 0
        labels_all: list[int] = []
        predictions_all: list[int] = []
        coverage = np.zeros(len(PILLAR_NAMES), dtype=np.float64)
        with torch.set_grad_enabled(training):
            for batch in loader:
                sequences, availability, reliability, labels = self._move_batch(batch)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                output = self.model(sequences, availability, reliability)
                loss, loss_terms = self._loss(output, labels)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    optimizer.step()
                batch_size = int(labels.shape[0])
                total_examples += batch_size
                total_loss += float(loss.detach().cpu()) * batch_size
                total_main_loss += loss_terms["main_loss"] * batch_size
                total_auxiliary_loss += loss_terms["auxiliary_loss"] * batch_size
                labels_all.extend(labels.detach().cpu().tolist())
                predictions_all.extend(output["predicted_class"].detach().cpu().tolist())
                coverage += output["fusion_mask"].detach().float().sum(dim=0).cpu().numpy()
        metrics = classification_metrics(labels_all, predictions_all)
        if total_examples:
            metrics.update(
                {
                    "loss": total_loss / total_examples,
                    "main_loss": total_main_loss / total_examples,
                    "auxiliary_loss": total_auxiliary_loss / total_examples,
                    "pillar_coverage_fraction": {
                        name: float(coverage[index] / total_examples)
                        for index, name in enumerate(PILLAR_NAMES)
                    },
                }
            )
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        """Evaluate with modality dropout disabled and report balanced metrics."""

        return self._run_epoch(loader, optimizer=None)

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        validation_metrics: Mapping[str, Any],
        training_metrics: Mapping[str, Any] | None = None,
        test_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            # These keys are intentionally stable: inference reconstructs the
            # exact model/extractor contract from them.
            "task": CHECKPOINT_TASK,
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "model_config": _model_config_payload(self.model.config),
            "extractor_config": _json_safe(asdict(self.extractor_config)),
            "extractor_schema_version": SCHEMA_VERSION,
            "normalizer": self.normalizer.state_dict(),
            "class_names": list(CLASS_NAMES),
            # Experiment protocol and diagnostics.
            "label_rule": LABEL_RULE,
            "official_split_codes": dict(OFFICIAL_SPLIT_CODES),
            "class_weights": None if self.class_weights is None else self.class_weights.detach().cpu().tolist(),
            "auxiliary_weight": self.auxiliary_weight,
            "validation_metrics": _json_safe(validation_metrics),
            "training_metrics": _json_safe(training_metrics) if training_metrics is not None else None,
            "test_metrics": _json_safe(test_metrics) if test_metrics is not None else None,
        }

    def save_checkpoint(
        self,
        epoch: int,
        validation_metrics: Mapping[str, Any],
        *,
        training_metrics: Mapping[str, Any] | None = None,
        test_metrics: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically write the current model as the canonical best checkpoint."""

        target = self.checkpoint_path
        _atomic_torch_save(
            target,
            self._checkpoint_payload(
                epoch=epoch,
                validation_metrics=validation_metrics,
                training_metrics=training_metrics,
                test_metrics=test_metrics,
            ),
        )
        return target

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        *,
        epochs: int = 25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 2,
        scheduler_factor: float = 0.5,
        min_learning_rate: float = 1e-5,
        early_stopping_patience: int | None = 8,
        early_stopping_min_delta: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Fit and retain the epoch with the highest validation macro F1.

        A validation-metric scheduler and patience stop prevent the compact
        LIRIS model from continuing far past its useful generalisation point.
        Set ``early_stopping_patience=None`` to run every requested epoch.
        """

        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if scheduler_patience < 0:
            raise ValueError("scheduler_patience must be zero or positive")
        if not 0.0 < scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be in (0, 1)")
        if min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be positive")
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")
        if early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_learning_rate,
        )
        best_macro_f1 = -float("inf")
        epochs_without_improvement = 0
        self.history = []
        self.stopped_early = False
        self.stop_reason = None
        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, optimizer=optimizer)
            validation_metrics = self._run_epoch(validation_loader, optimizer=None)
            macro_f1 = float(validation_metrics["macro_f1"])
            improved = macro_f1 > best_macro_f1 + early_stopping_min_delta
            if improved:
                best_macro_f1 = macro_f1
                epochs_without_improvement = 0
                self.best_epoch = epoch
                self.best_validation_metrics = dict(validation_metrics)
                self.save_checkpoint(epoch, validation_metrics, training_metrics=train_metrics)
            else:
                epochs_without_improvement += 1
            scheduler.step(macro_f1)
            entry: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "improved_validation_macro_f1": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
            self.history.append(entry)
            print(json.dumps(_json_safe(entry), sort_keys=True))
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                self.stopped_early = True
                self.stop_reason = (
                    "validation_macro_f1 did not improve by at least "
                    f"{early_stopping_min_delta:g} for {epochs_without_improvement} epoch(s)"
                )
                entry["early_stopping"] = self.stop_reason
                print(json.dumps(_json_safe({"early_stopping": self.stop_reason}), sort_keys=True))
                break
        return list(self.history)

    def add_test_metrics(self, test_metrics: Mapping[str, Any]) -> Path:
        """Attach held-out metrics to the already-selected best checkpoint.

        The method restores the best checkpoint state into the in-memory model
        first.  This avoids accidentally writing the last epoch instead of the
        validation-selected epoch after test evaluation.
        """

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("No best checkpoint exists; call fit before adding test metrics.")
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("task") != CHECKPOINT_TASK:
            raise ValueError("Checkpoint task does not match semantic five-pillar training.")
        payload["test_metrics"] = _json_safe(test_metrics)
        payload["training_history"] = _json_safe(self.history)
        _atomic_torch_save(self.checkpoint_path, payload)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        return self.checkpoint_path


def precompute_liris_semantic_cache(
    video_dir: str | Path,
    ranking_path: str | Path,
    sets_path: str | Path,
    cache_dir: str | Path,
    *,
    extractor_config: SemanticPillarConfig | None = None,
    limit_per_split: int | None = None,
    precompute_workers: int = 0,
    precompute_worker_threads: int | None = None,
    progress_every: int = 25,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Precompute each official split and expose extraction/audio diagnostics.

    Set ``precompute_workers`` to a small positive number (normally ``2`` for
    CPU ResNet extraction) to use spawned Windows worker processes.  The
    default remains serial for backwards compatibility.  Parallel mode never
    downloads model weights and therefore needs any requested ResNet weights to
    already be present in the local Torch cache.
    """

    datasets, records = build_liris_datasets(
        video_dir,
        ranking_path,
        sets_path,
        cache_dir,
        extractor_config=extractor_config,
        require_complete=require_complete,
    )
    return {
        "record_summary": records_summary(records),
        "splits": {
            split: dataset.precompute(
                limit_per_split,
                workers=precompute_workers,
                worker_threads=precompute_worker_threads,
                progress_every=progress_every,
            )
            for split, dataset in datasets.items()
        },
    }


def audit_liris_semantic_cache(
    video_dir: str | Path,
    ranking_path: str | Path,
    sets_path: str | Path,
    cache_dir: str | Path,
    *,
    extractor_config: SemanticPillarConfig | None = None,
    require_complete: bool = True,
    example_limit: int = 10,
) -> dict[str, Any]:
    """Read-only audit of every official split's semantic feature cache.

    Unlike ``precompute_liris_semantic_cache``, this never extracts frames or
    writes cache files.  It is intended for a quick integrity check before a
    run, and makes unexpected modality failures visible in saved experiment
    output rather than hiding them in a model metric.
    """

    datasets, records = build_liris_datasets(
        video_dir,
        ranking_path,
        sets_path,
        cache_dir,
        extractor_config=extractor_config,
        require_complete=require_complete,
    )
    return {
        "record_summary": records_summary(records),
        "splits": {
            split: dataset.audit_cache(example_limit=example_limit)
            for split, dataset in datasets.items()
        },
    }


def train_liris_semantic_five_pillar(
    video_dir: str | Path,
    ranking_path: str | Path,
    sets_path: str | Path,
    cache_dir: str | Path,
    checkpoint_dir: str | Path,
    *,
    extractor_config: SemanticPillarConfig | None = None,
    model_config: SemanticFivePillarConfig | None = None,
    batch_size: int = 16,
    num_workers: int = 0,
    epochs: int = 25,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    auxiliary_weight: float = 0.10,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    min_learning_rate: float = 1e-5,
    early_stopping_patience: int | None = 8,
    early_stopping_min_delta: float = 0.0,
    seed: int | None = DEFAULT_TRAINING_SEED,
    deterministic: bool = True,
    device: str | torch.device | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Run the full official-split semantic training/evaluation workflow.

    Normalisation is fit exclusively from training records.  The returned test
    metrics correspond to the checkpoint selected only by validation macro F1.
    This function does not download data or pretrained weights itself.
    """

    resolved_seed = seed_everything(seed, deterministic=deterministic)
    extractor_config = extractor_config or SemanticPillarConfig()
    expected_dims = SemanticPillarExtractor(extractor_config).pillar_dims
    
    loaders, datasets, records = build_liris_loaders(
        video_dir,
        ranking_path,
        sets_path,
        cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=resolved_seed,
        extractor_config=extractor_config,
        require_complete=require_complete,
    )
    cache_audit_before = {
        split: dataset.audit_cache()
        for split, dataset in datasets.items()
    }
    normalizer = PillarFeatureNormalizer.fit(datasets["train"])
    requested_model_config = model_config or compact_semantic_five_pillar_config(expected_dims)
    if dict(requested_model_config.input_dims) != dict(expected_dims):
        raise ValueError(
            "model_config.input_dims must match semantic extractor expected_dims; "
            f"got {dict(requested_model_config.input_dims)}, expected {dict(expected_dims)}."
        )
    active_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = SemanticFivePillarTrainer(
        SemanticFivePillarModel(requested_model_config),
        normalizer=normalizer,
        extractor_config=extractor_config,
        checkpoint_dir=checkpoint_dir,
        device=active_device,
        class_weights=compute_class_weights(records["train"], active_device),
        auxiliary_weight=auxiliary_weight,
    )
    history = trainer.fit(
        loaders["train"],
        loaders["validation"],
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        min_learning_rate=min_learning_rate,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
    )
    best_payload = torch.load(trainer.checkpoint_path, map_location=active_device, weights_only=False)
    trainer.model.load_state_dict(best_payload["model_state_dict"])
    trainer.model.to(active_device)
    test_metrics = trainer.evaluate(loaders["test"])
    checkpoint = trainer.add_test_metrics(test_metrics)
    cache_audit_after = {
        split: dataset.audit_cache()
        for split, dataset in datasets.items()
    }
    return {
        "checkpoint": str(checkpoint),
        "record_summary": records_summary(records),
        "best_epoch": trainer.best_epoch,
        "best_validation_metrics": trainer.best_validation_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "normalizer": normalizer.state_dict(),
        "model_config": _model_config_payload(requested_model_config),
        "cache_audit": {
            "before": cache_audit_before,
            "after": cache_audit_after,
        },
        "seed": resolved_seed,
        "deterministic": bool(deterministic),
        "epochs_completed": len(history),
        "stopped_early": trainer.stopped_early,
        "stop_reason": trainer.stop_reason,
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True, help="Directory containing raw LIRIS .mp4 clips.")
    parser.add_argument("--ranking-path", required=True, help="Path to ACCEDEranking.txt.")
    parser.add_argument("--sets-path", required=True, help="Path to ACCEDEsets.txt.")
    parser.add_argument("--cache-dir", required=True, help="Writable semantic feature-cache directory.")
    parser.add_argument("--checkpoint-dir", help="Writable directory for best_semantic_five_pillar.pt (train mode).")
    parser.add_argument("--mode", choices=("precompute", "audit", "train"), default="precompute")
    parser.add_argument("--limit-per-split", type=int, help="Optional small smoke-test limit per official split.")
    parser.add_argument(
        "--precompute-workers",
        type=int,
        default=0,
        help=(
            "Spawn this many CPU cache workers in precompute mode; 0 keeps serial extraction. "
            "Use 2 first for CPU ResNet extraction."
        ),
    )
    parser.add_argument(
        "--precompute-worker-threads",
        type=int,
        help=(
            "Optional Torch/OpenCV thread cap per cache worker. By default a conservative "
            "cap is derived from CPU count and worker count."
        ),
    )
    parser.add_argument(
        "--precompute-progress-every",
        type=int,
        default=25,
        help="Print cache progress after this many clips; 0 disables periodic progress lines.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--auxiliary-weight", type=float, default=0.10)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Experiment seed. Use --no-deterministic only when reproducibility is not required.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="Keep the seed but permit nondeterministic Torch kernels.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop after this many non-improving validation epochs; use -1 to disable.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--embedding-dim", type=int, default=int(DEFAULT_COMPACT_MODEL_KWARGS["embedding_dim"]))
    parser.add_argument("--attention-dim", type=int, default=int(DEFAULT_COMPACT_MODEL_KWARGS["attention_dim"]))
    parser.add_argument(
        "--temporal-hidden-dim",
        type=int,
        default=int(DEFAULT_COMPACT_MODEL_KWARGS["temporal_hidden_dim"]),
    )
    parser.add_argument(
        "--fusion-hidden-dim",
        type=int,
        default=int(DEFAULT_COMPACT_MODEL_KWARGS["fusion_hidden_dim"]),
    )
    parser.add_argument("--dropout", type=float, default=float(DEFAULT_COMPACT_MODEL_KWARGS["dropout"]))
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=float(DEFAULT_COMPACT_MODEL_KWARGS["modality_dropout"]),
    )
    parser.add_argument("--device", help="Torch device, for example cuda or cpu.")
    parser.add_argument("--allow-partial-data", action="store_true", help="Allow missing annotated videos for a deliberate smoke test.")
    parser.add_argument("--sampled-windows", type=int, default=16)
    parser.add_argument("--max-duration-seconds", type=float, default=15.0)
    parser.add_argument("--use-pretrained-spatial", action="store_true", help="Use a locally cached ResNet-18 context encoder.")
    parser.add_argument("--allow-model-download", action="store_true", help="Permit torchvision to download ResNet-18 weights if absent.")
    parser.add_argument("--spatial-device", default="cpu")
    parser.add_argument("--use-panns", action="store_true", help="Use PANNs CNN14 for audio semantics.")
    parser.add_argument("--use-clip", action="store_true", help="Use CLIP ViT-B/32 for spatial context.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; defaults to safe cache precomputation, not training."""

    args = _build_cli_parser().parse_args(argv)
    extractor_config = SemanticPillarConfig(
        max_duration_seconds=args.max_duration_seconds,
        sampled_windows=args.sampled_windows,
        use_pretrained_spatial=args.use_pretrained_spatial,
        allow_model_download=args.allow_model_download,
        spatial_device=args.spatial_device,
        use_panns=args.use_panns,
        use_clip=args.use_clip,
    )
    common = {
        "video_dir": args.video_dir,
        "ranking_path": args.ranking_path,
        "sets_path": args.sets_path,
        "cache_dir": args.cache_dir,
        "extractor_config": extractor_config,
        "require_complete": not args.allow_partial_data,
    }
    if args.mode == "precompute":
        report = precompute_liris_semantic_cache(
            **common,
            limit_per_split=args.limit_per_split,
            precompute_workers=args.precompute_workers,
            precompute_worker_threads=args.precompute_worker_threads,
            progress_every=args.precompute_progress_every,
        )
    elif args.mode == "audit":
        report = audit_liris_semantic_cache(**common)
    else:
        if not args.checkpoint_dir:
            raise SystemExit("--checkpoint-dir is required in --mode train.")
        model_config = compact_semantic_five_pillar_config(
            SemanticPillarExtractor(extractor_config).pillar_dims,
            embedding_dim=args.embedding_dim,
            attention_dim=args.attention_dim,
            temporal_hidden_dim=args.temporal_hidden_dim,
            fusion_hidden_dim=args.fusion_hidden_dim,
            dropout=args.dropout,
            modality_dropout=args.modality_dropout,
        )
        report = train_liris_semantic_five_pillar(
            **common,
            checkpoint_dir=args.checkpoint_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            auxiliary_weight=args.auxiliary_weight,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            min_learning_rate=args.min_learning_rate,
            early_stopping_patience=(
                None if args.early_stopping_patience < 0 else args.early_stopping_patience
            ),
            early_stopping_min_delta=args.early_stopping_min_delta,
            seed=args.seed,
            deterministic=not args.no_deterministic,
            model_config=model_config,
            device=args.device,
        )
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    raise SystemExit(main())


__all__ = [
    "CACHE_FORMAT_VERSION",
    "CHECKPOINT_TASK",
    "CLASS_NAMES",
    "DEFAULT_COMPACT_MODEL_KWARGS",
    "DEFAULT_TRAINING_SEED",
    "LABEL_RULE",
    "LirisRecord",
    "LirisSemanticFivePillarDataset",
    "OFFICIAL_SPLIT_CODES",
    "PillarFeatureNormalizer",
    "SemanticFeatureLoadError",
    "SemanticFivePillarTrainer",
    "TOTAL_LIRIS_RANKS",
    "audit_liris_semantic_cache",
    "build_liris_datasets",
    "build_liris_loaders",
    "build_liris_records",
    "classification_metrics",
    "compact_semantic_five_pillar_config",
    "compute_class_weights",
    "main",
    "precompute_liris_semantic_cache",
    "rank_to_class",
    "records_summary",
    "seed_everything",
    "semantic_five_pillar_collate",
    "train_liris_semantic_five_pillar",
]
