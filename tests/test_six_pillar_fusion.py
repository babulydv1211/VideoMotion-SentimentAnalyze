"""Behavioral tests for the explicit spatial-anchor fusion policy."""

from __future__ import annotations

import numpy as np

from six_pillar_fusion import CLASS_NAMES, UNIFORM, PillarResult, fuse_six_pillars


def _pillar(
    name: str,
    probs: list[float],
    *,
    reliability: float = 1.0,
    available: bool = True,
    contributes_to_valence: bool = True,
    status: str = "available",
    base_weight: float | None = None,
    evidence: dict | None = None,
) -> dict:
    """Use a plain mapping to verify the public integration contract."""

    return {
        "name": name,
        "probs": probs,
        "class_order": CLASS_NAMES,
        "reliability": reliability,
        "available": available,
        "contributes_to_valence": contributes_to_valence,
        "status": status,
        "base_weight": base_weight,
        "evidence": evidence or {},
    }


def test_clear_spatial_anchor_is_the_final_decision_and_explains_weights():
    result = fuse_six_pillars(
        [
            _pillar("Spatial / trained CLIP", [0.08, 0.12, 0.80], status="spatial_anchor", reliability=0.9),
            _pillar("Speech", [0.20, 0.10, 0.70]),
            _pillar("Acoustic", [0.20, 0.20, 0.60]),
            _pillar("Color", [0.30, 0.30, 0.40], reliability=0.5),
            _pillar("Motion", UNIFORM.tolist(), contributes_to_valence=False, status="structural_only"),
            _pillar("Temporal", UNIFORM.tolist(), contributes_to_valence=False, status="structural_only"),
        ]
    )

    assert result.spatial_is_anchor
    assert result.status == "anchored"
    assert result.label == "Positive"
    assert result.decision_label == "Positive"
    assert CLASS_NAMES[int(np.argmax(result.probabilities))] == "Positive"
    assert result.contributions["Spatial / trained CLIP"].effective_weight >= 0.70
    assert result.contributions["Motion"].state == "structural_only"
    assert result.contributions["Motion"].effective_weight == 0.0
    np.testing.assert_allclose(result.probabilities.sum(), 1.0)


def test_near_tied_spatial_is_provisional_or_uncertain_even_if_other_pillars_vote():
    result = fuse_six_pillars(
        [
            _pillar("Spatial", [0.34, 0.33, 0.33], status="visual_uncertain", reliability=0.9),
            _pillar("Speech", [0.02, 0.02, 0.96]),
            _pillar("Acoustic", [0.10, 0.10, 0.80]),
        ]
    )

    assert not result.spatial_is_anchor
    assert result.status in {"provisional", "uncertain"}
    assert result.decision_label.startswith("Provisional:") or result.decision_label == "Uncertain"
    assert result.contributions["Spatial"].state == "spatial_provisional"


def test_noisy_audio_cannot_overturn_a_clear_spatial_anchor():
    """Two fully Negative audio rules cannot silently flip a Positive visual anchor."""

    positive_anchor = fuse_six_pillars(
        [
            # A lead of 0.12 is enough under an explicit trained-head anchor
            # status, but raw weighted evidence can be challenged by every
            # supporting source.  Protection must retain Positive.
            _pillar("Spatial", [0.40, 0.08, 0.52], status="spatial_anchor", reliability=0.95),
            _pillar("Speech", [1.0, 0.0, 0.0]),
            _pillar("Acoustic", [1.0, 0.0, 0.0]),
            _pillar("Color", [1.0, 0.0, 0.0]),
        ]
    )

    assert positive_anchor.label == "Positive"
    assert positive_anchor.decision_label == "Positive"
    assert CLASS_NAMES[int(np.argmax(positive_anchor.probabilities))] == "Positive"
    assert positive_anchor.status == "anchored_with_conflict"
    assert positive_anchor.anchor_protection_applied


def test_strong_opposing_audio_is_exposed_as_conflict_not_hidden():
    result = fuse_six_pillars(
        [
            _pillar("Spatial", [0.05, 0.10, 0.85], status="spatial_anchor", reliability=0.9),
            _pillar("Speech", [0.90, 0.05, 0.05]),
            _pillar("Acoustic", [0.34, 0.33, 0.33]),  # weak/non-conflicting
        ]
    )

    assert result.status == "anchored_with_conflict"
    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.pillar == "Speech"
    assert conflict.spatial_label == "Positive"
    assert conflict.pillar_label == "Negative"
    assert conflict.severity == "strong"


def test_all_uniform_votes_are_safe_and_never_invent_a_label():
    result = fuse_six_pillars(
        [
            PillarResult("Spatial", UNIFORM, reliability=0.0, available=False, status="no_visual_frames"),
            _pillar("Speech", UNIFORM.tolist(), available=False, reliability=0.0, status="no_audio"),
            _pillar("Acoustic", UNIFORM.tolist(), available=False, reliability=0.0, status="no_audio"),
            _pillar("Color", UNIFORM.tolist(), reliability=0.0, status="low_color_information"),
            _pillar("Motion", UNIFORM.tolist(), contributes_to_valence=False),
            _pillar("Temporal", UNIFORM.tolist(), contributes_to_valence=False),
        ]
    )

    np.testing.assert_allclose(result.probabilities, UNIFORM)
    np.testing.assert_allclose(result.evidence_probabilities, UNIFORM)
    assert result.status == "uncertain"
    assert result.decision_label == "Uncertain"
    assert all(item.effective_weight == 0.0 for item in result.contributions.values())
