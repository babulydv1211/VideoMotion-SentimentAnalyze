"""Focused contract tests for the offline full-ResNet visual baseline."""

from __future__ import annotations

import joblib
import numpy as np

from scene_motion_llm.training.resnet18_visual_baseline import (
    RESNET18_EMBEDDING_DIM,
    ResNet18FeatureCache,
    ResNet18FeatureConfig,
    ResNet18FeatureResult,
    ResNet18Record,
    fit_train_only_pca,
    train_resnet18_pca_logistic_baseline,
    uniform_frame_indices,
)


def _result(value: float, *, frames: int = 3) -> ResNet18FeatureResult:
    embedding = np.full((frames, RESNET18_EMBEDDING_DIM), value, dtype=np.float32)
    # Give PCA a small time-varying signal without changing the cache contract.
    embedding[:, 0] += np.arange(frames, dtype=np.float32)
    return ResNet18FeatureResult(
        embeddings=embedding,
        frame_indices=np.arange(frames, dtype=np.int64),
        timestamps_seconds=np.arange(frames, dtype=np.float32) / 24.0,
        video_metadata={"fps": 24.0},
    )


def _record(root, split: str, label: int, index: int) -> ResNet18Record:
    source = root / f"{split}_{label}_{index}.mp4"
    # The cache validates source fingerprints but does not need to decode in
    # these unit tests.  Production extraction always uses genuine raw video.
    source.write_bytes(f"source-{split}-{label}-{index}".encode("utf-8"))
    return ResNet18Record(
        video_id=source.stem,
        video_path=str(source),
        label=label,
        split=split,
    )


def test_uniform_sampling_is_ordered_unique_and_bounded():
    assert uniform_frame_indices(1, 16).tolist() == [0]
    indices = uniform_frame_indices(100, 16)
    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] == 99
    assert np.all(np.diff(indices) > 0)


def test_feature_cache_rejects_a_changed_raw_source(tmp_path):
    config = ResNet18FeatureConfig()
    cache = ResNet18FeatureCache(tmp_path / "cache", config)
    record = _record(tmp_path, "train", 0, 0)
    cache.store(record, _result(1.0), model_metadata={"weights": "test-local"})

    loaded = cache.load(record)
    assert loaded is not None
    assert loaded.embeddings.shape == (3, RESNET18_EMBEDDING_DIM)
    np.testing.assert_allclose(loaded.embeddings, _result(1.0).embeddings)

    # Size changes in addition to timestamp, so this is robust on file systems
    # with coarse timestamp precision.
    (tmp_path / "train_0_0.mp4").write_bytes(b"a materially changed source video")
    assert cache.load(record) is None


def test_pca_mean_uses_training_frames_only():
    train_sequences = [_result(1.0).embeddings, _result(3.0).embeddings]
    reducer = fit_train_only_pca(train_sequences, pca_components=2, seed=7)
    expected_mean = np.concatenate(train_sequences, axis=0).mean(axis=0)

    np.testing.assert_allclose(reducer.mean_, expected_mean)
    assert reducer.n_components_ == 2


def test_compact_baseline_selects_on_validation_and_persists_train_only_pca(tmp_path):
    config = ResNet18FeatureConfig()
    cache = ResNet18FeatureCache(tmp_path / "cache", config)
    records_by_split = {"train": [], "validation": [], "test": []}
    train_sequences: list[np.ndarray] = []

    # Each class has a stable separate block.  Test values are deliberately
    # shifted far away; checking the saved PCA mean below proves they were not
    # included in the reducer fit.
    for split, count, shift in (("train", 4, 0.0), ("validation", 1, 100.0), ("test", 1, 1_000.0)):
        for label in range(3):
            for index in range(count):
                record = _record(tmp_path, split, label, index)
                result = _result(shift + label * 10.0 + index * 0.01)
                cache.store(record, result, model_metadata={"weights": "test-local"})
                records_by_split[split].append(record)
                if split == "train":
                    train_sequences.append(result.embeddings)

    report = train_resnet18_pca_logistic_baseline(
        records_by_split,
        tmp_path / "cache",
        tmp_path / "output",
        feature_config=config,
        pca_components=4,
        c_values=(0.1, 1.0),
        seed=11,
        max_iter=500,
    )

    assert report["protocol"]["pca_fit_split"] == "train_frames_only"
    assert report["protocol"]["selection_split"] == "validation"
    assert report["split_sizes"] == {"train": 12, "validation": 3, "test": 3}
    assert (tmp_path / "output" / "resnet18_pca_logistic_baseline.json").is_file()
    payload = joblib.load(tmp_path / "output" / "resnet18_pca_logistic_baseline.joblib")
    expected_train_mean = np.concatenate(train_sequences, axis=0).mean(axis=0)
    np.testing.assert_allclose(payload["pca"].mean_, expected_train_mean)
