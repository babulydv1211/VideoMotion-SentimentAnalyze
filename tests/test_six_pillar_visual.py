import numpy as np

from six_pillar_visual import run_color_pillar, spatial_result_from_frame_scores


def test_near_tied_spatial_scores_are_visible_but_not_anchor_eligible():
    result = spatial_result_from_frame_scores(
        np.array([[0.34, 0.33, 0.33], [0.34, 0.33, 0.33]], dtype=np.float32)
    )

    assert result["available"] is True
    assert result["status"] == "visual_uncertain"
    assert result["evidence"]["anchor_eligible"] is False
    np.testing.assert_allclose(result["probs"].sum(), 1.0)


def test_clear_consistent_spatial_scores_are_anchor_eligible():
    result = spatial_result_from_frame_scores(
        np.array([[0.10, 0.12, 0.32], [0.11, 0.11, 0.34], [0.12, 0.10, 0.33]], dtype=np.float32)
    )

    assert result["status"] == "spatial_anchor"
    assert result["evidence"]["top_label"] == "Positive"
    assert result["evidence"]["anchor_eligible"] is True


def test_color_is_bounded_and_grayscale_abstains():
    grayscale = np.full((120, 160, 3), 110, dtype=np.uint8)
    grayscale_result = run_color_pillar([grayscale])
    assert grayscale_result["status"] == "low_color_information"
    np.testing.assert_allclose(grayscale_result["probs"], np.full(3, 1 / 3))

    bright = np.full((120, 160, 3), [240, 220, 80], dtype=np.uint8)
    bright_result = run_color_pillar([bright])
    assert bright_result["status"] == "weak_lighting_context"
    assert float(bright_result["probs"].max() - bright_result["probs"].min()) < 0.20
