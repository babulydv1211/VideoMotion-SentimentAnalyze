"""Small deterministic checks for the canonical LIRIS training contract."""

from __future__ import annotations

from pathlib import Path
import random
import tempfile

import numpy as np
import pytest
import torch

from scene_motion_llm.models.semantic_five_pillar import PILLAR_NAMES, SemanticFivePillarModel
from scene_motion_llm.training.semantic_five_pillar_training import (
    LirisRecord,
    LirisSemanticFivePillarDataset,
    PillarFeatureNormalizer,
    SemanticFeatureLoadError,
    SemanticFivePillarTrainer,
    build_liris_records,
    compact_semantic_five_pillar_config,
    rank_to_class,
    seed_everything,
    semantic_five_pillar_collate,
)
from scene_motion_llm.utils.semantic_pillars import PILLAR_DIMS, SemanticPillarConfig


def test_rank_thirds_keep_the_positive_neutral_negative_class_order():
    assert rank_to_class(0) == 2
    assert rank_to_class(3265) == 2
    assert rank_to_class(3266) == 1
    assert rank_to_class(6532) == 1
    assert rank_to_class(6533) == 0
    assert rank_to_class(9799) == 0


def test_official_split_codes_join_to_the_expected_named_splits():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        videos = root / "videos"
        videos.mkdir()
        names = ("ACCEDE00000.mp4", "ACCEDE00001.mp4", "ACCEDE00002.mp4")
        for name in names:
            (videos / name).touch()
        ranking = root / "ranking.tsv"
        ranking.write_text(
            "id\tname\tvalenceRank\n"
            "0\tACCEDE00000.mp4\t0\n"
            "1\tACCEDE00001.mp4\t3266\n"
            "2\tACCEDE00002.mp4\t6533\n",
            encoding="utf-8",
        )
        sets = root / "sets.tsv"
        sets.write_text(
            "id\tname\tset\n"
            "0\tACCEDE00000.mp4\t1\n"
            "1\tACCEDE00001.mp4\t2\n"
            "2\tACCEDE00002.mp4\t0\n",
            encoding="utf-8",
        )

        records = build_liris_records(videos, ranking, sets)

    assert records["train"][0].label == 2
    assert records["validation"][0].label == 1
    assert records["test"][0].label == 0


def test_normalizer_does_not_create_audio_values_for_a_mute_clip():
    statistics = {
        name: {"mean": [2.0] * PILLAR_DIMS[name], "std": [4.0] * PILLAR_DIMS[name]}
        for name in PILLAR_NAMES
    }
    normalizer = PillarFeatureNormalizer(statistics)
    sequences = {
        name: torch.zeros(1, 2, PILLAR_DIMS[name], dtype=torch.float32)
        for name in PILLAR_NAMES
    }
    availability = {
        name: torch.ones(1, 2, dtype=torch.float32)
        for name in PILLAR_NAMES
    }
    reliability = {
        name: torch.ones(1, 2, dtype=torch.float32)
        for name in PILLAR_NAMES
    }
    availability["audio"].zero_()
    reliability["audio"].zero_()

    normalized = normalizer.apply(sequences, availability, reliability)

    assert torch.count_nonzero(normalized["audio"]) == 0
    assert torch.allclose(normalized["color"], torch.full_like(normalized["color"], -0.5))


def test_seed_everything_replays_python_numpy_and_torch_rngs():
    seed_everything(814, deterministic=False)
    first = (random.random(), float(np.random.random()), torch.rand(3))
    seed_everything(814, deterministic=False)
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_compact_training_config_is_well_below_the_legacy_capacity():
    config = compact_semantic_five_pillar_config(PILLAR_DIMS)
    model = SemanticFivePillarModel(config)

    assert config.embedding_dim == 48
    assert config.fusion_hidden_dim == 96
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000


def test_collate_and_normalizer_accept_an_extractor_with_dynamic_feature_dimensions():
    dimensions = {name: PILLAR_DIMS[name] + 3 for name in PILLAR_NAMES}
    batch = []
    for index, length in enumerate((2, 3)):
        batch.append(
            {
                "sequences": {
                    name: torch.full((length, dimensions[name]), float(index))
                    for name in PILLAR_NAMES
                },
                "availability": {name: torch.ones(length) for name in PILLAR_NAMES},
                "reliability": {name: torch.ones(length) for name in PILLAR_NAMES},
                "label": index,
                "video_id": f"clip-{index}",
            }
        )
    collated = semantic_five_pillar_collate(batch)
    statistics = {
        name: {"mean": [0.0] * dimensions[name], "std": [1.0] * dimensions[name]}
        for name in PILLAR_NAMES
    }
    normalized = PillarFeatureNormalizer(statistics).apply(
        collated["sequences"], collated["availability"], collated["reliability"]
    )

    assert collated["sequences"]["audio"].shape == (2, 3, dimensions["audio"])
    assert normalized["spatial"].shape[-1] == dimensions["spatial"]


def test_dataset_failure_is_reported_for_its_own_record_without_substitution():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        record = LirisRecord(
            video_id="missing",
            video_path=str(root / "missing.mp4"),
            valence_rank=10,
            label=2,
            split="train",
        )
        dataset = LirisSemanticFivePillarDataset([record], root / "cache")
        audit = dataset.audit_cache()
        with pytest.raises(SemanticFeatureLoadError, match="missing"):
            _ = dataset[0]

    assert audit["source_errors"] == 1
    assert audit["all_current"] is False


def test_trainer_stops_after_configured_non_improving_validation_epochs():
    statistics = {
        name: {"mean": [0.0] * PILLAR_DIMS[name], "std": [1.0] * PILLAR_DIMS[name]}
        for name in PILLAR_NAMES
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = SemanticFivePillarTrainer(
            SemanticFivePillarModel(compact_semantic_five_pillar_config(PILLAR_DIMS)),
            normalizer=PillarFeatureNormalizer(statistics),
            extractor_config=SemanticPillarConfig(),
            checkpoint_dir=temp_dir,
            device="cpu",
        )

        def constant_metrics(_loader, optimizer=None):
            return {"macro_f1": 0.4 if optimizer is None else 0.3}

        trainer._run_epoch = constant_metrics  # type: ignore[method-assign]
        history = trainer.fit(
            object(),
            object(),
            epochs=5,
            early_stopping_patience=1,
        )

    assert len(history) == 2
    assert trainer.best_epoch == 1
    assert trainer.stopped_early is True
