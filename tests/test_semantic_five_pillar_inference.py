"""Focused contract tests for semantic five-pillar upload inference."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scene_motion_llm.inference.semantic_five_pillar_inference import (
    CheckpointContractError,
    SemanticFivePillarInferencer,
)
from scene_motion_llm.models.semantic_five_pillar import (
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


class _StaticExtractor:
    def __init__(self, result):
        self.result = result

    def extract(self, _path):
        return self.result


def _checkpoint(path: Path, *, nested_normalizer: bool = False) -> None:
    torch.manual_seed(7)
    config = SemanticFivePillarConfig(embedding_dim=16, attention_dim=8, temporal_hidden_dim=12, fusion_hidden_dim=24)
    model = SemanticFivePillarModel(config)
    normalizer = {
        pillar: {
            "mean": [0.0] * dimension,
            "std": [1.0] * dimension,
        }
        for pillar, dimension in PILLAR_DIMS.items()
    }
    saved_normalizer = (
        {"format": "pillar-feature-normalizer-v1", "epsilon": 1e-6, "pillars": normalizer}
        if nested_normalizer
        else normalizer
    )
    torch.save(
        {
            "task": "liris_semantic_five_pillar",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "extractor_config": asdict(SemanticPillarConfig()),
            "extractor_schema_version": SCHEMA_VERSION,
            "normalizer": saved_normalizer,
        },
        path,
    )


def _checkpoint_for_extractor_config(
    path: Path,
    extractor_config: SemanticPillarConfig,
    *,
    schema_version: str = SCHEMA_VERSION,
) -> None:
    """Write a tiny checkpoint with the exact dynamic extractor contract."""

    torch.manual_seed(13)
    input_dims = SemanticPillarExtractor(extractor_config).pillar_dims
    config = SemanticFivePillarConfig(
        input_dims=input_dims,
        embedding_dim=16,
        attention_dim=8,
        temporal_hidden_dim=12,
        fusion_hidden_dim=24,
    )
    model = SemanticFivePillarModel(config)
    normalizer = {
        pillar: {"mean": [0.0] * dimension, "std": [1.0] * dimension}
        for pillar, dimension in input_dims.items()
    }
    torch.save(
        {
            "task": "liris_semantic_five_pillar",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "extractor_config": asdict(extractor_config),
            "extractor_schema_version": schema_version,
            "normalizer": normalizer,
        },
        path,
    )


def _extraction_for_dims(dims: dict[str, int], steps: int = 3):
    features = {
        pillar: np.full((steps, dimension), 0.25, dtype=np.float32)
        for pillar, dimension in dims.items()
    }
    masks = {pillar: np.ones(steps, dtype=np.float32) for pillar in PILLAR_NAMES}
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor": {"config": {"pillar_dims": dict(dims)}},
        "features": features,
        "availability": masks,
        "reliability": dict(masks),
    }


def _mute_extraction(steps: int = 4):
    rng = np.random.default_rng(3)
    features = {
        pillar: rng.normal(size=(steps, dimension)).astype(np.float32)
        for pillar, dimension in PILLAR_DIMS.items()
    }
    features["audio"] = np.zeros((steps, PILLAR_DIMS["audio"]), dtype=np.float32)
    availability = {pillar: np.ones(steps, dtype=np.float32) for pillar in PILLAR_NAMES}
    reliability = {pillar: np.full(steps, 0.9, dtype=np.float32) for pillar in PILLAR_NAMES}
    availability["audio"][:] = 0.0
    reliability["audio"][:] = 0.0
    # Simulate a black/flat visual source: it remains observable, but with
    # weak information quality rather than a fabricated affect class.
    reliability["color"][:] = 0.05
    reliability["spatial"][:] = 0.05
    labels = {
        "audio": [["audio_unavailable"] for _ in range(steps)],
        "color": [["low_visual_detail"] for _ in range(steps)],
        "spatial": [["low_spatial_detail"] for _ in range(steps)],
        "motion": [["moderate_motion"] for _ in range(steps)],
        "temporal": [["gradual_visual_change"] for _ in range(steps)],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "video": {"duration_seconds": 10.0},
        "features": features,
        "availability": availability,
        "reliability": reliability,
        "evidence": {
            "audio": {"audio_status": "silent_audio_track"},
            "color": {"note": "low detail"},
            "spatial": {"note": "low detail"},
            "motion": {"note": "kinematic cue"},
            "temporal": {"note": "pacing cue"},
            "notes": {"reliability": "low quality is not a sentiment label"},
        },
        "semantic_labels": labels,
        "temporal_inputs": {"timestamps_seconds": np.linspace(0.0, 9.5, steps, dtype=np.float32)},
    }


class SemanticFivePillarInferenceTest(unittest.TestCase):
    def test_mute_audio_is_nil_zero_weight_and_low_detail_is_not_a_negative_rule(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "semantic.pt"
            _checkpoint(checkpoint_path)
            inferencer = SemanticFivePillarInferencer(checkpoint_path, device="cpu")
            inferencer.extractor = _StaticExtractor(_mute_extraction())

            result = inferencer.analyze_video("ignored-by-static-extractor.mp4")

        self.assertIn(result["sentiment"], {"Positive", "Neutral", "Negative"})
        self.assertTrue(np.isfinite(result["confidence"]))
        self.assertEqual(result["pillars"]["audio"]["status"], "NIL")
        self.assertEqual(result["audio_status"], "silent_audio_track")
        self.assertEqual(result["fusion_weights"]["audio"], 0.0)
        self.assertEqual(result["pillar_weights"]["audio"], 0.0)
        self.assertIn("evidence", result)
        self.assertTrue(
            any("does not imply Negative sentiment" in warning for warning in result["warnings"])
        )

    def test_rejects_legacy_checkpoint_before_any_video_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy.pt"
            torch.save({"task": "legacy_raw_video", "model_state_dict": {}}, checkpoint_path)
            with self.assertRaises(CheckpointContractError):
                SemanticFivePillarInferencer(checkpoint_path, device="cpu")

    def test_accepts_the_normalizer_state_written_by_the_canonical_trainer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "semantic.pt"
            _checkpoint(checkpoint_path, nested_normalizer=True)
            inferencer = SemanticFivePillarInferencer(checkpoint_path, device="cpu")

        self.assertEqual(tuple(inferencer.class_names), ("Positive", "Neutral", "Negative"))

    def test_optional_dimensions_are_dynamic_and_unavailable_audio_keeps_its_width(self):
        baseline = SemanticPillarExtractor(SemanticPillarConfig())
        panns = SemanticPillarExtractor(SemanticPillarConfig(use_panns=True))
        clip = SemanticPillarExtractor(SemanticPillarConfig(use_clip=True))

        self.assertEqual(panns.pillar_dims["audio"], 547)
        self.assertEqual(clip.pillar_dims["spatial"], 528)
        self.assertEqual(len(panns.pillar_features["audio"]), 547)
        self.assertEqual(len(clip.pillar_features["spatial"]), 528)
        self.assertNotEqual(baseline.cache_signature(), panns.cache_signature())
        self.assertNotEqual(baseline.cache_signature(), clip.cache_signature())
        self.assertIn("implementation", baseline.config_metadata())
        self.assertEqual(clip.config_metadata()["pillar_dims"]["spatial"], 528)

        # No librosa/audio decoder is needed to exercise the unavailable path.
        panns._librosa = lambda: None  # type: ignore[method-assign]
        values, availability, reliability, _, _ = panns._audio_pillar(
            Path("unused.mp4"),
            np.asarray([0.0, 1.0], dtype=np.float32),
            2.0,
        )
        self.assertEqual(values.shape, (2, 547))
        self.assertEqual(availability.shape, (2,))
        self.assertEqual(reliability.shape, (2,))

    def test_inference_uses_optional_checkpoint_dimensions_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "semantic_optional.pt"
            extractor_config = SemanticPillarConfig(use_panns=True, use_clip=True)
            _checkpoint_for_extractor_config(checkpoint_path, extractor_config)
            inferencer = SemanticFivePillarInferencer(checkpoint_path, device="cpu")
            dims = SemanticPillarExtractor(extractor_config).pillar_dims
            sequences, _, _, _ = inferencer._prepare_model_inputs(_extraction_for_dims(dims))

        self.assertEqual(tuple(sequences["audio"].shape), (1, 3, 547))
        self.assertEqual(tuple(sequences["spatial"].shape), (1, 3, 528))

    def test_v2_baseline_checkpoint_is_allowed_but_v2_optional_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_path = root / "v2_baseline.pt"
            _checkpoint_for_extractor_config(
                baseline_path,
                SemanticPillarConfig(),
                schema_version=LEGACY_BASELINE_SCHEMA_VERSION,
            )
            inferencer = SemanticFivePillarInferencer(baseline_path, device="cpu")
            self.assertEqual(inferencer.checkpoint_schema_version, LEGACY_BASELINE_SCHEMA_VERSION)

            optional_path = root / "v2_optional.pt"
            _checkpoint_for_extractor_config(
                optional_path,
                SemanticPillarConfig(use_panns=True),
                schema_version=LEGACY_BASELINE_SCHEMA_VERSION,
            )
            with self.assertRaises(CheckpointContractError):
                SemanticFivePillarInferencer(optional_path, device="cpu")


if __name__ == "__main__":
    unittest.main()
