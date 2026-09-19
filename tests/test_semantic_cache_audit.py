from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scene_motion_llm.scripts.audit_semantic_cache import _has_integrity_problem, audit_cache


def _write_pair(directory: Path, stem: str, *, signature: str = "signature") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{stem}.npz",
        availability_audio=np.ones(2, dtype=np.float32),
    )
    (directory / f"{stem}.json").write_text(
        json.dumps(
            {
                "extractor_signature": signature,
                "extractor_schema_version": "semantic-five-pillar-v2",
                "audio": {"audio_status": "audio_present"},
            }
        ),
        encoding="utf-8",
    )


def test_audit_accepts_a_consistent_minimal_cache(tmp_path: Path):
    for split in ("train", "validation", "test"):
        _write_pair(tmp_path / split, "clip")

    report = audit_cache(tmp_path)

    assert report["splits"]["train"]["pairs"] == 1
    assert report["splits"]["test"]["audio_availability_mean"] == 1.0
    assert not _has_integrity_problem(report)
    assert not _has_integrity_problem(report, min_audio_availability=0.95)


def test_audit_detects_cross_split_signature_and_orphan(tmp_path: Path):
    _write_pair(tmp_path / "train", "clip", signature="one")
    _write_pair(tmp_path / "validation", "clip", signature="one")
    _write_pair(tmp_path / "test", "clip", signature="two")
    np.savez_compressed(tmp_path / "test" / "orphan.npz", availability_audio=np.ones(2, dtype=np.float32))

    report = audit_cache(tmp_path)

    assert report["splits"]["test"]["orphan_npz"] == 1
    assert _has_integrity_problem(report)


def test_audit_can_enforce_expected_audio_coverage(tmp_path: Path):
    for split in ("train", "validation", "test"):
        _write_pair(tmp_path / split, "clip")
    np.savez_compressed(tmp_path / "test" / "clip.npz", availability_audio=np.zeros(2, dtype=np.float32))

    report = audit_cache(tmp_path)

    assert not _has_integrity_problem(report)
    assert _has_integrity_problem(report, min_audio_availability=0.95)
