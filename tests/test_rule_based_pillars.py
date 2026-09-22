"""Model-free checks for the rule-based Streamlit application's pillars.

The application is intentionally loaded with a tiny Streamlit stand-in so these
tests exercise the production functions without starting a Streamlit runtime or
initialising/downloading CLIP, Whisper, or the text-classification model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


class _Container:
    """Minimal context manager returned by Streamlit layout calls."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _CacheDecorator:
    """Imitates the part of Streamlit's cache API used during import."""

    def __call__(self, function=None, **_kwargs):
        if function is None:
            return lambda decorated: decorated
        return function

    def clear(self):
        return None


def _streamlit_stub() -> ModuleType:
    """Create only the Streamlit surface reached before an upload is selected."""

    stub = ModuleType("streamlit")
    stub.cache_resource = _CacheDecorator()
    stub.cache_data = _CacheDecorator()
    stub.set_page_config = lambda *_args, **_kwargs: None
    stub.markdown = lambda *_args, **_kwargs: None
    stub.title = lambda *_args, **_kwargs: None
    stub.caption = lambda *_args, **_kwargs: None
    stub.button = lambda *_args, **_kwargs: False
    stub.file_uploader = lambda *_args, **_kwargs: None

    def columns(spec, **_kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return tuple(_Container() for _ in range(count))

    stub.columns = columns
    return stub


@pytest.fixture(scope="module")
def rule_app():
    """Load ``app.py`` without launching its UI or loading any ML weights."""

    module_name = "_rule_based_app_under_test"
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    original_streamlit = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _streamlit_stub()

    spec = importlib.util.spec_from_file_location(module_name, app_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if original_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = original_streamlit


def _assert_probability_distribution(probabilities: np.ndarray) -> None:
    assert probabilities.shape == (3,)
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=0, atol=1e-6)


def test_abstaining_inputs_do_not_invoke_model_loaders(rule_app, monkeypatch):
    """Missing media and empty frames should stay local and return neutral votes."""

    def model_access_is_forbidden(*_args, **_kwargs):
        raise AssertionError("a model loader was called for an abstaining input")

    monkeypatch.setattr(rule_app, "load_clip", model_access_is_forbidden)
    monkeypatch.setattr(rule_app, "load_stt_model", model_access_is_forbidden)
    monkeypatch.setattr(rule_app, "load_emotion_model", model_access_is_forbidden)

    expected = np.full(3, 1 / 3, dtype=np.float32)
    clip_vote = rule_app.run_clip_zeroshot([])
    speech_vote, transcript = rule_app.run_speech_sentiment(None)
    acoustic_vote = rule_app.run_acoustic_emotion(None)

    np.testing.assert_allclose(clip_vote, expected)
    np.testing.assert_allclose(speech_vote, expected)
    np.testing.assert_allclose(acoustic_vote, expected)
    assert transcript == "[No audio track found]"


def test_color_pillar_abstains_for_grayscale_footage(rule_app):
    """A grayscale palette is an explicit aesthetic-filter abstention case."""

    ramp = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    grayscale_frame = np.dstack([ramp, ramp, ramp])

    vote = rule_app.run_color([grayscale_frame])

    np.testing.assert_allclose(vote, np.full(3, 1 / 3, dtype=np.float32))


def test_static_motion_returns_neutral(rule_app):
    """Identical frames result in a neutral motion vote."""

    frame = np.full((96, 96, 3), 127, dtype=np.uint8)
    vote = rule_app.run_motion([frame, frame.copy()])

    _assert_probability_distribution(vote)
    np.testing.assert_allclose(vote, [0, 1, 0], atol=1e-2)


def test_temporal_pillar_reacts_to_cuts_and_stability(rule_app):
    """Editing choices contribute heuristic sentiment signals."""

    bright = np.full((128, 128, 3), 255, dtype=np.uint8)
    dark = np.zeros((128, 128, 3), dtype=np.uint8)
    stable_vote = rule_app.run_temporal([bright, bright.copy(), bright.copy()])
    cuts_vote = rule_app.run_temporal([bright, dark, bright])

    _assert_probability_distribution(stable_vote)
    _assert_probability_distribution(cuts_vote)
    
    np.testing.assert_allclose(stable_vote, [0, 0.7, 0.3], atol=1e-2)
    np.testing.assert_allclose(cuts_vote, [0.4, 0.6, 0], atol=1e-2)


def test_fusion_normalizes_fixed_weights_and_mutes_all_uniform_votes(rule_app):
    """Uniform votes abstain; usable signals retain their relative base weights."""

    uniform = np.full(3, 1 / 3, dtype=np.float32)
    pillars = [
        (np.array([0.98, 0.01, 0.01], dtype=np.float32), 0.40, "Spatial"),
        (uniform, 0.10, "Speech"),
        (np.array([0.01, 0.01, 0.98], dtype=np.float32), 0.25, "Acoustic"),
        (uniform, 0.05, "Color"),
        (uniform, 0.15, "Motion"),
        (uniform, 0.05, "Temporal"),
    ]

    fused, weights = rule_app.fuse_pillars(pillars)

    _assert_probability_distribution(fused)
    assert weights["Speech"] == 0.0
    assert weights["Color"] == 0.0
    assert weights["Motion"] == 0.0
    assert weights["Temporal"] == 0.0
    np.testing.assert_allclose(sum(weights.values()), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(weights["Spatial"], 0.40 / 0.65, rtol=0, atol=1e-6)
    np.testing.assert_allclose(weights["Acoustic"], 0.25 / 0.65, rtol=0, atol=1e-6)


def test_fusion_does_not_amplify_a_confident_pillar(rule_app):
    """A sharp speech vote keeps its configured weight instead of taking over."""

    pillars = [
        (np.array([0.34, 0.33, 0.33], dtype=np.float32), 0.40, "Spatial"),
        (np.array([0.001, 0.001, 0.998], dtype=np.float32), 0.10, "Speech"),
        (np.array([0.44, 0.15, 0.41], dtype=np.float32), 0.25, "Acoustic"),
        (np.array([0.38, 0.23, 0.39], dtype=np.float32), 0.05, "Color"),
        (np.array([0.31, 0.37, 0.32], dtype=np.float32), 0.15, "Motion"),
        (np.array([0.30, 0.40, 0.30], dtype=np.float32), 0.05, "Temporal"),
    ]

    _, weights = rule_app.fuse_pillars(pillars)

    assert weights == {
        "Spatial": 0.40,
        "Speech": 0.10,
        "Acoustic": 0.25,
        "Color": 0.05,
        "Motion": 0.15,
        "Temporal": 0.05,
    }


def test_close_fusion_scores_are_reported_as_mixed(rule_app):
    """A modest lead is a mixed result, while a clear lead remains decisive."""

    assert rule_app.is_mixed_result(np.array([0.35, 0.24, 0.39], dtype=np.float32))
    assert not rule_app.is_mixed_result(np.array([0.32, 0.24, 0.44], dtype=np.float32))
    assert not rule_app.is_mixed_result(np.array([0.15, 0.20, 0.65], dtype=np.float32))


@pytest.mark.parametrize(
    ("probs", "expected_label"),
    [
        (np.array([0.405, 0.395, 0.20], dtype=np.float32), "Mixed: Negative / Neutral"),
        (np.array([0.405, 0.15, 0.395], dtype=np.float32), "Mixed: Negative / Positive"),
        (np.array([0.20, 0.395, 0.405], dtype=np.float32), "Mixed: Positive / Neutral"),
    ],
)
def test_mixed_result_names_the_two_close_classes(rule_app, probs, expected_label):
    """Mixed output explains which pair is tied instead of hiding it."""

    is_mixed, display_label, _runner_up_idx, _margin = rule_app.summarize_fusion_result(probs)

    assert is_mixed
    assert display_label == expected_label


def test_three_close_scores_name_all_three_classes(rule_app):
    """A three-way tie should not be reduced to an arbitrary pair."""

    is_mixed, display_label, _runner_up_idx, _margin = rule_app.summarize_fusion_result(
        np.array([0.34, 0.33, 0.33], dtype=np.float32)
    )

    assert is_mixed
    assert display_label == "Mixed: Negative / Neutral / Positive"


def test_exact_two_way_tie_does_not_imply_a_leader(rule_app):
    """Exact ties retain a stable class order rather than inventing a winner."""

    is_mixed, display_label, _runner_up_idx, _margin = rule_app.summarize_fusion_result(
        np.array([0.40, 0.40, 0.20], dtype=np.float32)
    )

    assert is_mixed
    assert display_label == "Mixed: Negative / Neutral"


def test_reported_accede_00636_votes_keep_a_meaningful_positive_lead(rule_app):
    """A leading class need not exceed 50% in a three-way fusion."""

    uniform = np.full(3, 1 / 3, dtype=np.float32)
    pillars = [
        (np.array([0.32, 0.36, 0.33], dtype=np.float32), 0.40, "Spatial"),
        (np.array([0.00, 0.00, 1.00], dtype=np.float32), 0.10, "Speech"),
        (np.array([0.44, 0.15, 0.41], dtype=np.float32), 0.25, "Acoustic"),
        (np.array([0.38, 0.23, 0.38], dtype=np.float32), 0.05, "Color"),
        (uniform, 0.15, "Motion"),
        (uniform, 0.05, "Temporal"),
    ]

    fused, weights = rule_app.fuse_pillars(pillars)

    np.testing.assert_allclose(fused, [0.32125, 0.24125, 0.441875], atol=1e-6)
    assert weights["Speech"] == 0.125
    assert not rule_app.is_mixed_result(fused)


def test_reported_accede_00649_near_tie_remains_mixed(rule_app):
    """The observed 1.3-point Positive/Neutral gap should stay uncertain."""

    mixed, display_label, _runner_up_idx, _margin = rule_app.summarize_fusion_result(
        np.array([0.204, 0.392, 0.405], dtype=np.float32)
    )

    assert mixed
    assert display_label == "Mixed: Positive / Neutral"


def test_fusion_rejects_an_invalid_total_weight(rule_app):
    """A clear error is safer than emitting NaN scores for unusable inputs."""

    with pytest.raises(ValueError, match="positive finite total"):
        rule_app.fuse_pillars(
            [(np.full(3, 1 / 3, dtype=np.float32), 0.0, "Speech")]
        )


def test_extract_media_caps_audio_to_the_frame_window(rule_app, monkeypatch):
    """Audio extraction must cover the same bounded interval as sampled frames."""

    class FakeAudioSegment:
        def __init__(self):
            self.closed = False
            self.output_path = None

        def write_audiofile(self, filename, **_kwargs):
            self.output_path = filename

        def close(self):
            self.closed = True

    class FakeAudio:
        def __init__(self):
            self.subclip_calls = []
            self.segment = FakeAudioSegment()

        def subclip(self, start, end):
            self.subclip_calls.append((start, end))
            return self.segment

    class FakeClip:
        duration = 60.0
        fps = 30.0

        def __init__(self):
            self.audio = FakeAudio()
            self.closed = False

        def get_frame(self, _time):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    clip = FakeClip()
    monkeypatch.setattr(rule_app, "VideoFileClip", lambda _path: clip)

    frames, wav_path = rule_app.extract_media("unit-test-upload.mp4")

    assert len(frames) == rule_app.SAMPLED_FRAMES
    assert clip.audio.subclip_calls == [(0, rule_app.MAX_DURATION)]
    assert clip.audio.segment.closed
    assert clip.closed
    assert wav_path == "unit-test-upload.mp4_audio.wav"
    assert clip.audio.segment.output_path == wav_path


@pytest.mark.parametrize(
    "transcript",
    [
        "[No audio track found]",
        "[Audio too short]",
        "[No speech detected]",
        "unintelligible words [Gibberish Detected — Muting (20% lexical validity)]",
        "hello [Low Confidence: Muted]",
        "[Error: unavailable model]",
    ],
)
def test_speech_statuses_are_not_displayed_as_transcripts(rule_app, transcript):
    """Abstention/error messages must not be presented as heard speech."""

    assert rule_app.speech_has_status(transcript)
    assert not rule_app.speech_has_status("a normal transcribed sentence")


def test_media_extraction_trims_audio_to_the_frame_window(rule_app, monkeypatch):
    """Long inputs must not send their full soundtrack to Whisper or Librosa."""

    class FakeAudio:
        def __init__(self):
            self.window = None
            self.written_path = None

        def subclipped(self, start, end):
            self.window = (start, end)
            return self

        def write_audiofile(self, path, **_kwargs):
            self.written_path = path

    class FakeVideoClip:
        def __init__(self, _path):
            self.duration = 60.0
            self.fps = 10.0
            self.audio = FakeAudio()
            self.closed = False

        def get_frame(self, _timestamp):
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    clip = FakeVideoClip("ignored")
    monkeypatch.setattr(rule_app, "MAX_DURATION", 3.0)
    monkeypatch.setattr(rule_app, "VideoFileClip", lambda _path: clip)

    frames, wav_path = rule_app.extract_media("unit-test-clip.mp4")

    assert len(frames) == rule_app.SAMPLED_FRAMES
    assert clip.audio.window == (0, 3.0)
    assert clip.audio.written_path == wav_path
    assert clip.closed
