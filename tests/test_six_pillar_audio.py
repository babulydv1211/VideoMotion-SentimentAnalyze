"""Focused unit tests for the conservative six-pillar audio contract.

Every model and decoder is injected/mocked: these tests never load Whisper,
Transformers, or a network resource.
"""

from __future__ import annotations

import numpy as np

import six_pillar_audio as audio


class _KnownWords:
    def __init__(self, words: set[str] | None = None) -> None:
        self.words = words

    def known(self, words):
        values = set(words)
        return values if self.words is None else values & self.words


class _Whisper:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def transcribe(self, path, fp16=False):
        self.calls.append((path, fp16))
        return self.result


def _audio_loader(samples: np.ndarray, sample_rate: int):
    def load(_path, *, sr, mono):
        assert mono is True
        assert sr == sample_rate
        return samples, sample_rate

    return load


def _wav_path(tmp_path):
    path = tmp_path / "clip.wav"
    path.touch()
    return path


def _good_segments():
    return [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "i feel happy today",
            "no_speech_prob": 0.03,
            "avg_logprob": -0.20,
            "compression_ratio": 1.10,
        }
    ]


def test_speech_rejects_whisper_no_speech_metadata(tmp_path):
    rate = 16_000
    samples = np.full(rate * 2, 0.05, dtype=np.float32)
    whisper = _Whisper(
        {
            "text": "this sounds like a complete sentence today",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "this sounds like a complete sentence today",
                    "no_speech_prob": 0.95,
                    "avg_logprob": -0.25,
                    "compression_ratio": 1.05,
                }
            ],
        }
    )

    result = audio.run_speech_pillar(
        _wav_path(tmp_path),
        whisper,
        lambda _text: [{"label": "joy", "score": 1.0}],
        spellchecker=_KnownWords(),
        audio_loader=_audio_loader(samples, rate),
    )

    assert result.name == "Speech"
    assert not result.available
    assert result.reason == "whisper_no_speech"
    np.testing.assert_allclose(result.probs, [1 / 3, 1 / 3, 1 / 3])


def test_speech_smooths_a_text_model_extreme_using_input_quality(tmp_path):
    rate = 16_000
    time = np.arange(rate * 2, dtype=np.float32) / rate
    samples = 0.08 * np.sin(2 * np.pi * 180 * time)
    whisper = _Whisper(
        {
            "text": "i feel very happy today and love this scene",
            "segments": _good_segments(),
        }
    )

    result = audio.run_speech_pillar(
        _wav_path(tmp_path),
        whisper,
        lambda _text: [[{"label": "joy", "score": 0.99}, {"label": "love", "score": 0.01}]],
        spellchecker=_KnownWords(),
        audio_loader=_audio_loader(samples, rate),
    )

    assert result.name == "Speech"
    assert result.available
    assert result.status == "used"
    assert result.evidence["transcript"].startswith("i feel")
    assert 0.40 <= result.reliability <= 1.0
    # The injected model was effectively 100% positive; the pillar must never
    # forward that as a 100% fusion vote.
    assert result.probs[2] < 0.80
    assert result.probs[2] > result.probs[0]
    np.testing.assert_allclose(result.probs.sum(), 1.0)


def test_speech_requires_lexically_plausible_transcript(tmp_path):
    rate = 16_000
    samples = np.full(rate * 2, 0.05, dtype=np.float32)
    whisper = _Whisper(
        {
            "text": "qzxw plok asdf zzzq blorb",
            "segments": _good_segments(),
        }
    )

    result = audio.run_speech_pillar(
        _wav_path(tmp_path),
        whisper,
        lambda _text: [{"label": "joy", "score": 1.0}],
        spellchecker=_KnownWords({"as"}),
        audio_loader=_audio_loader(samples, rate),
    )

    assert not result.available
    assert result.reason == "low_lexical_validity"
    assert result.evidence["lexical_validity"] < 0.70


def test_acoustic_abstains_on_silence(tmp_path):
    rate = 22_050
    result = audio.run_acoustic_pillar(
        _wav_path(tmp_path),
        audio_loader=_audio_loader(np.zeros(rate, dtype=np.float32), rate),
    )

    assert result.name == "Acoustic"
    assert not result.available
    assert result.reason == "insufficient_active_audio"


def test_acoustic_abstains_on_ambiguous_music_without_voice(tmp_path, monkeypatch):
    rate = 22_050
    time = np.arange(rate * 2, dtype=np.float32) / rate
    samples = 0.12 * np.sin(2 * np.pi * 220 * time)

    monkeypatch.setattr(
        audio.librosa,
        "pyin",
        lambda *_args, **_kwargs: (
            np.full(16, np.nan),
            np.zeros(16, dtype=bool),
            np.zeros(16, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(audio.librosa.effects, "hpss", lambda y: (y, np.zeros_like(y)))
    monkeypatch.setattr(audio.librosa.feature, "spectral_flatness", lambda **_kwargs: np.array([[0.01]]))

    result = audio.run_acoustic_pillar(
        _wav_path(tmp_path),
        audio_loader=_audio_loader(samples, rate),
    )

    assert not result.available
    assert result.reason == "music_ambiguous"
    assert result.evidence["music_ambiguity"] >= 0.60


def test_acoustic_voiced_result_is_bounded_not_an_extreme(tmp_path, monkeypatch):
    rate = 22_050
    time = np.arange(rate * 2, dtype=np.float32) / rate
    # Amplitude changes ensure a non-uniform but still very weak DSP cue.
    samples = (0.06 + 0.04 * np.sin(2 * np.pi * 2 * time)) * np.sin(2 * np.pi * 180 * time)
    monkeypatch.setattr(
        audio.librosa,
        "pyin",
        lambda *_args, **_kwargs: (
            np.full(20, 180.0),
            np.ones(20, dtype=bool),
            np.ones(20, dtype=np.float32),
        ),
    )

    result = audio.run_acoustic_pillar(
        _wav_path(tmp_path),
        audio_loader=_audio_loader(samples, rate),
    )

    assert result.name == "Acoustic"
    assert result.available
    assert result.status == "used"
    assert result.reliability >= 0.35
    assert np.max(result.probs) < 0.65
    np.testing.assert_allclose(result.probs.sum(), 1.0)
