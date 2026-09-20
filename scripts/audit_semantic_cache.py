"""Audit semantic five-pillar cache integrity without mutating it.

The canonical training path uses cached ``.npz`` arrays together with JSON
metadata.  This command makes stale, incomplete, or modality-inconsistent
caches visible before a model is trained or its test score is reported.

Example::

    python -m scene_motion_llm.scripts.audit_semantic_cache \
        --cache-dir .\\scene_motion_llm\\cache\\semantic_five_pillar_v2 --strict
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SPLITS = ("train", "validation", "test")


def _read_metadata(path: Path) -> Mapping[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _audio_availability(npz_path: Path) -> float | None:
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            values = np.asarray(payload["availability_audio"], dtype=np.float32)
    except (OSError, ValueError, KeyError):
        return None
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        return None
    return float(values.mean())


def audit_cache(cache_dir: str | Path) -> dict[str, Any]:
    """Return a JSON-safe integrity summary for each canonical split.

    The function deliberately does not attempt a repair.  A cache mismatch
    should be repaired by rerunning the canonical precompute command with the
    same extractor configuration used for the intended experiment.
    """

    root = Path(cache_dir)
    result: dict[str, Any] = {"cache_dir": str(root), "splits": {}}
    for split in SPLITS:
        directory = root / split
        npz_paths = {path.stem: path for path in directory.glob("*.npz")} if directory.is_dir() else {}
        json_paths = {path.stem: path for path in directory.glob("*.json")} if directory.is_dir() else {}
        all_stems = sorted(set(npz_paths) | set(json_paths))
        statuses: Counter[str] = Counter()
        signatures: Counter[str] = Counter()
        schemas: Counter[str] = Counter()
        invalid_metadata: list[str] = []
        invalid_audio_arrays: list[str] = []
        audio_means: list[float] = []

        for stem in all_stems:
            npz_path = npz_paths.get(stem)
            json_path = json_paths.get(stem)
            if npz_path is None:
                invalid_metadata.append(f"{stem}:missing_npz")
                continue
            if json_path is None:
                invalid_metadata.append(f"{stem}:missing_json")
                continue
            metadata = _read_metadata(json_path)
            if metadata is None:
                invalid_metadata.append(f"{stem}:invalid_json")
                continue
            signatures[str(metadata.get("extractor_signature", "missing"))] += 1
            schemas[str(metadata.get("extractor_schema_version", "missing"))] += 1
            audio = metadata.get("audio", {})
            if isinstance(audio, Mapping):
                statuses[str(audio.get("audio_status", "unknown"))] += 1
            else:
                statuses["invalid_audio_metadata"] += 1
            availability = _audio_availability(npz_path)
            if availability is None:
                invalid_audio_arrays.append(stem)
            else:
                audio_means.append(availability)

        result["splits"][split] = {
            "pairs": len(set(npz_paths) & set(json_paths)),
            "orphan_npz": len(set(npz_paths) - set(json_paths)),
            "orphan_json": len(set(json_paths) - set(npz_paths)),
            "invalid_metadata": invalid_metadata[:20],
            "invalid_metadata_count": len(invalid_metadata),
            "invalid_audio_arrays": invalid_audio_arrays[:20],
            "invalid_audio_array_count": len(invalid_audio_arrays),
            "audio_status_counts": dict(sorted(statuses.items())),
            "audio_availability_mean": None if not audio_means else float(np.mean(audio_means)),
            "extractor_signatures": dict(sorted(signatures.items())),
            "schema_versions": dict(sorted(schemas.items())),
        }
    return result


def _has_integrity_problem(
    report: Mapping[str, Any], *, min_audio_availability: float | None = None
) -> bool:
    splits = report.get("splits", {})
    if not isinstance(splits, Mapping):
        return True
    signatures: set[str] = set()
    schemas: set[str] = set()
    for details in splits.values():
        if not isinstance(details, Mapping):
            return True
        if int(details.get("orphan_npz", 0)) or int(details.get("orphan_json", 0)):
            return True
        if int(details.get("invalid_metadata_count", 0)) or int(details.get("invalid_audio_array_count", 0)):
            return True
        availability = details.get("audio_availability_mean")
        if (
            min_audio_availability is not None
            and (not isinstance(availability, (int, float)) or float(availability) < min_audio_availability)
        ):
            return True
        signatures.update(dict(details.get("extractor_signatures", {})))
        schemas.update(dict(details.get("schema_versions", {})))
    return len(signatures) > 1 or len(schemas) > 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Root containing train/validation/test cache folders.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when an integrity issue is found.")
    parser.add_argument(
        "--min-audio-availability",
        type=float,
        help="Optional per-split minimum mean audio availability; useful for detecting unexpected decode regressions.",
    )
    args = parser.parse_args(argv)
    if args.min_audio_availability is not None and not 0.0 <= args.min_audio_availability <= 1.0:
        parser.error("--min-audio-availability must be in [0, 1].")
    report = audit_cache(args.cache_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and _has_integrity_problem(
        report,
        min_audio_availability=args.min_audio_availability,
    ) else 0


if __name__ == "__main__":  # pragma: no cover - command-line convenience
    raise SystemExit(main())
