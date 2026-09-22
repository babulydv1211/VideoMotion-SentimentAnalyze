"""Focused tests for conservative structural six-pillar measurements."""

from __future__ import annotations

import cv2
import numpy as np

from six_pillar_structural import measure_motion, measure_temporal


UNIFORM = np.full(3, 1.0 / 3.0, dtype=np.float32)


def _textured_frame(height: int = 120, width: int = 160) -> np.ndarray:
    """Make deterministic, flow-friendly BGR content without external assets."""

    y, x = np.mgrid[:height, :width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = ((3 * x + 5 * y) % 256).astype(np.uint8)
    image[..., 1] = ((7 * x + 2 * y) % 256).astype(np.uint8)
    image[..., 2] = ((11 * x + 13 * y) % 256).astype(np.uint8)
    cv2.circle(image, (width // 3, height // 2), 18, (255, 255, 255), -1)
    return image


def _translated_frames() -> list[np.ndarray]:
    base = _textured_frame()
    frames = []
    for offset in range(6):
        matrix = np.float32([[1, 0, 3 * offset], [0, 1, 0]])
        frames.append(cv2.warpAffine(base, matrix, (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REFLECT))
    return frames


def test_motion_measures_global_translation_but_abstains_from_valence() -> None:
    result = measure_motion(_translated_frames())

    assert result["available"] is True
    assert result["status"] == "measured"
    assert result["contributes_to_valence"] is False
    np.testing.assert_allclose(result["probs"], UNIFORM)
    assert result["reliability"] > 0.2
    assert result["evidence"]["global_motion_px"] > 1.0
    assert result["evidence"]["global_motion_fraction"] > 0.30
    assert result["evidence"]["valid_pairs"] == 5


def test_motion_separates_local_object_motion_from_global_translation() -> None:
    base = _textured_frame()
    frames: list[np.ndarray] = []
    for offset in range(6):
        frame = base.copy()
        cv2.rectangle(frame, (15 + 12 * offset, 40), (55 + 12 * offset, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (18 + 12 * offset, 43), (52 + 12 * offset, 82), (255, 255, 255), -1)
        frames.append(frame)

    result = measure_motion(frames)
    evidence = result["evidence"]

    assert result["available"] is True
    assert evidence["local_motion_px"] > evidence["global_motion_px"]
    assert evidence["localized_motion_fraction"] > evidence["global_motion_fraction"]


def test_temporal_measures_hard_cut_and_pacing_without_a_sentiment_vote() -> None:
    dark = _textured_frame()
    dark = (dark.astype(np.float32) * 0.15).astype(np.uint8)
    bright = _textured_frame()
    bright = np.clip(bright.astype(np.int16) + 150, 0, 255).astype(np.uint8)
    result = measure_temporal([dark, dark.copy(), bright, bright.copy()], fps=2.0)

    assert result["available"] is True
    assert result["status"] == "measured"
    assert result["contributes_to_valence"] is False
    np.testing.assert_allclose(result["probs"], UNIFORM)
    assert result["evidence"]["cut_count"] >= 1
    assert result["evidence"]["frame_difference_mean"] > 0.10
    assert result["evidence"]["pacing_index"] > 0.0
    assert result["evidence"]["cut_rate_per_second"] > 0.0


def test_structural_pillars_report_unavailable_for_one_frame() -> None:
    one_frame = _textured_frame()

    for result in (measure_motion([one_frame]), measure_temporal([one_frame])):
        assert result["available"] is False
        assert result["status"] == "insufficient_frames"
        assert result["reliability"] == 0.0
        assert result["contributes_to_valence"] is False
        np.testing.assert_allclose(result["probs"], UNIFORM)
