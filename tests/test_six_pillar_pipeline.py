import numpy as np

from six_pillar_audio import AudioPillarResult
from six_pillar_pipeline import _audio_mapping, _structural_frames_from_rgb


def test_audio_result_is_preserved_in_common_pillar_schema():
    audio = AudioPillarResult(
        probs=np.array([0.2, 0.4, 0.4], dtype=np.float32),
        reliability=0.6,
        available=True,
        status="used",
        reason="quality_gated_transcript",
        evidence={"transcript": "hello world there"},
    )
    result = _audio_mapping("Speech", audio)

    assert result["name"] == "Speech"
    assert result["contributes_to_valence"] is True
    assert result["evidence"]["reason"] == "quality_gated_transcript"
    np.testing.assert_allclose(result["probs"].sum(), 1.0)


def test_pipeline_converts_uploaded_rgb_to_structural_bgr():
    red_rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    red_rgb[..., 0] = 255
    converted = _structural_frames_from_rgb([red_rgb])[0]

    assert converted[0, 0].tolist() == [0, 0, 255]
