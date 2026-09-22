"""Conservative, inspectable audio pillars for the six-pillar demo.

The root application originally used a transcript classifier and a handful of
audio measurements as if either could be a decisive sentiment label.  This
module keeps the same ingredients, but gives callers a richer contract:

* unusable evidence explicitly *abstains* instead of emitting a fake vote;
* Whisper's own segment diagnostics gate a transcript before it reaches BERT;
* text-model distributions are blended back toward uniform according to input
  quality, so an imperfect transcript cannot become a 100% label; and
* acoustic DSP is treated as a weak prosody cue only when there is active,
  plausibly voiced audio.  Ambiguous music/ambient sound abstains.

There are deliberately no model-loading functions here.  Callers inject the
already-cached Whisper model and Hugging Face pipeline, which keeps this module
easy to test and prevents surprise downloads during import.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
import math
import re
from typing import Any, Callable, Mapping, Sequence

import librosa
import numpy as np


CLASS_NAMES: tuple[str, str, str] = ("Negative", "Neutral", "Positive")
_UNIFORM = np.full(3, 1.0 / 3.0, dtype=np.float32)

# Whisper's documented decoding defaults are useful *quality gates*, not a
# claim that all speech under these values is nonsense.  We use them to abstain
# rather than force a text-emotion prediction through a weak transcript.
_MAX_NO_SPEECH_PROB = 0.60
_MIN_AVG_LOGPROB = -1.00
_MAX_COMPRESSION_RATIO = 2.40


@dataclass(frozen=True)
class AudioPillarResult:
    """Output shared by the Speech and Acoustic pillars.

    ``probs`` always follows ``(Negative, Neutral, Positive)`` and sums to one.
    An unavailable result always uses the uniform distribution, which makes it
    safe for legacy callers that use uniformity as an abstention marker.

    ``reliability`` is intentionally separate from probability concentration:
    a BERT classifier can be highly concentrated while the transcript feeding
    it is poor.  Future fusion should use both ``available`` and reliability.
    """

    probs: np.ndarray
    reliability: float
    available: bool
    status: str
    reason: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    # These fields make the result directly consumable by six_pillar_fusion.
    # The audio functions set them to their actual signal; defaults keep the
    # class useful for callers constructing a generic audio result by hand.
    name: str = "Audio"
    contributes_to_valence: bool = True
    base_weight: float | None = None

    def __post_init__(self) -> None:
        probs = _normalise_probs(self.probs)
        if not self.available:
            probs = _UNIFORM.copy()
        probs.setflags(write=False)
        object.__setattr__(self, "probs", probs)
        object.__setattr__(self, "reliability", float(np.clip(self.reliability, 0.0, 1.0)))
        object.__setattr__(self, "evidence", dict(self.evidence))

    def as_dict(self) -> dict[str, Any]:
        """Return a UI/JSON-friendly copy of this result."""

        return {
            "probs": {
                label: float(self.probs[index]) for index, label in enumerate(CLASS_NAMES)
            },
            "reliability": self.reliability,
            "available": self.available,
            "status": self.status,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "name": self.name,
            "contributes_to_valence": self.contributes_to_valence,
            "base_weight": self.base_weight,
        }


def _normalise_probs(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a finite three-class distribution, falling back to uniform."""

    try:
        probs = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return _UNIFORM.copy()
    if probs.size != 3 or not np.all(np.isfinite(probs)):
        return _UNIFORM.copy()
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        return _UNIFORM.copy()
    return (probs / total).astype(np.float32)


def _abstain(
    reason: str,
    *,
    status: str = "abstained",
    evidence: Mapping[str, Any] | None = None,
    name: str = "Audio",
    base_weight: float | None = None,
) -> AudioPillarResult:
    return AudioPillarResult(
        probs=_UNIFORM.copy(),
        reliability=0.0,
        available=False,
        status=status,
        reason=reason,
        evidence=evidence or {},
        name=name,
        base_weight=base_weight,
    )


def _for_signal(result: AudioPillarResult, name: str, base_weight: float) -> AudioPillarResult:
    """Give a common result its concrete fusion identity."""

    return replace(result, name=name, base_weight=base_weight, contributes_to_valence=True)


def _signal_result(name: str, base_weight: float) -> Callable[[Callable[..., AudioPillarResult]], Callable[..., AudioPillarResult]]:
    """Decorate a pillar implementation with its stable fusion identity."""

    def decorate(function: Callable[..., AudioPillarResult]) -> Callable[..., AudioPillarResult]:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> AudioPillarResult:
            return _for_signal(function(*args, **kwargs), name, base_weight)

        return wrapped

    return decorate


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        converted = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return default
    return converted if math.isfinite(converted) else default


def _load_mono_audio(
    wav_path: str | Path | None,
    *,
    sample_rate: int,
    audio_loader: Callable[..., tuple[np.ndarray, int]] | None,
) -> tuple[np.ndarray, int] | AudioPillarResult:
    """Load finite mono audio, returning a structured abstention on failure."""

    if not wav_path:
        return _abstain("no_audio")
    path = Path(wav_path)
    if not path.is_file():
        return _abstain("no_audio", evidence={"path": str(path)})

    loader = audio_loader or librosa.load
    try:
        samples, loaded_rate = loader(str(path), sr=sample_rate, mono=True)
    except Exception as exc:  # Decoders surface library-specific exception types.
        return _abstain("audio_decode_failed", status="error", evidence={"error": str(exc)})

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    audio = audio.reshape(-1)
    if audio.size == 0:
        return _abstain("empty_audio")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    rate = int(loaded_rate or sample_rate)
    if rate <= 0:
        return _abstain("invalid_sample_rate")
    return audio, rate


def _rms_and_activity(audio: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Return frame RMS, active-frame share, RMS mean, and RMS variation."""

    frame_length = min(2048, max(64, int(audio.size)))
    hop_length = min(512, max(16, frame_length // 4))
    try:
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    except Exception:
        # A tiny fallback keeps a malformed analysis backend from turning audio
        # into a confident prediction.
        rms = np.array([float(np.sqrt(np.mean(np.square(audio))))], dtype=np.float32)

    rms = np.asarray(rms, dtype=np.float64)
    rms = rms[np.isfinite(rms)]
    if rms.size == 0:
        return np.array([], dtype=np.float64), 0.0, 0.0, 0.0

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_peak = float(np.quantile(rms, 0.95))
    # Relative threshold admits quiet but real speech; the absolute floor rejects
    # numerical noise / silent WAV files.
    active_threshold = max(3e-4, rms_peak * 0.20)
    active_fraction = float(np.mean(rms >= active_threshold)) if rms_peak >= 3e-4 else 0.0
    variation = rms_std / (rms_mean + 1e-8)
    return rms, active_fraction, rms_mean, float(variation)


def _speech_words(text: str) -> list[str]:
    """Keep alphabetic words while allowing normal English contractions."""

    return [word.lower() for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)]


def _get_spellchecker(spellchecker: Any | None) -> Any | None:
    if spellchecker is not None:
        return spellchecker
    try:
        from spellchecker import SpellChecker

        return SpellChecker()
    except Exception:
        return None


def _lexical_validity(words: Sequence[str], spellchecker: Any) -> float:
    """Compute a conservative, but not name-hostile, lexical-validity score."""

    if not words:
        return 0.0
    # SpellChecker expects plain word forms.  Keep a contraction's base form as
    # an alternate rather than declaring ordinary English contractions invalid.
    candidates = [word.replace("'", "") for word in words]
    try:
        known = {str(item).lower() for item in spellchecker.known(candidates)}
    except Exception:
        return 0.0
    recognized = sum(candidate in known for candidate in candidates)
    return recognized / len(candidates)


def _segment_metrics(segments: Any) -> tuple[dict[str, float | int | bool], str | None]:
    """Aggregate Whisper diagnostics with duration/text-length weights."""

    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)) or not segments:
        return {}, "missing_whisper_segments"

    weighted: list[tuple[float, float, float, float]] = []
    # rows are (weight, no_speech_prob, avg_logprob, compression_ratio)
    complete_rows = 0
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        no_speech = _safe_float(segment.get("no_speech_prob"))
        logprob = _safe_float(segment.get("avg_logprob"))
        compression = _safe_float(segment.get("compression_ratio"))
        if no_speech is None or logprob is None or compression is None:
            continue
        start = _safe_float(segment.get("start"), 0.0) or 0.0
        end = _safe_float(segment.get("end"), start) or start
        duration = max(0.0, end - start)
        text_weight = len(str(segment.get("text", "")).strip().split())
        weight = max(duration, float(text_weight), 1.0)
        weighted.append((weight, no_speech, logprob, compression))
        complete_rows += 1

    if not weighted:
        return {"segment_count": len(segments), "diagnostic_segments": 0}, "missing_whisper_diagnostics"

    total = sum(row[0] for row in weighted)
    no_speech = sum(row[0] * row[1] for row in weighted) / total
    logprob = sum(row[0] * row[2] for row in weighted) / total
    compression = sum(row[0] * row[3] for row in weighted) / total
    logprob_quality = _clip01((logprob - (-1.35)) / 1.15)
    compression_quality = _clip01((_MAX_COMPRESSION_RATIO + 0.45 - compression) / 0.75)
    whisper_quality = _clip01(
        0.45 * (1.0 - _clip01(no_speech))
        + 0.40 * logprob_quality
        + 0.15 * compression_quality
    )
    return {
        "segment_count": len(segments),
        "diagnostic_segments": complete_rows,
        "no_speech_prob": float(no_speech),
        "avg_logprob": float(logprob),
        "compression_ratio": float(compression),
        "whisper_quality": whisper_quality,
    }, None


def _call_transcriber(whisper_model: Any, wav_path: str) -> Mapping[str, Any]:
    if whisper_model is None:
        raise RuntimeError("Whisper model was not supplied")
    transcribe = getattr(whisper_model, "transcribe", whisper_model if callable(whisper_model) else None)
    if not callable(transcribe):
        raise TypeError("Whisper model must provide a callable transcribe method")
    try:
        result = transcribe(wav_path, fp16=False)
    except TypeError:
        # Test doubles and alternate wrappers often do not expose fp16.
        result = transcribe(wav_path)
    if not isinstance(result, Mapping):
        raise TypeError("Whisper transcribe result must be a mapping")
    return result


def _flatten_classifier_scores(raw_scores: Any) -> list[Mapping[str, Any]]:
    """Normalise common Hugging Face pipeline response shapes."""

    if isinstance(raw_scores, Mapping):
        return [raw_scores]
    if not isinstance(raw_scores, Sequence) or isinstance(raw_scores, (str, bytes)):
        return []
    if len(raw_scores) == 1 and isinstance(raw_scores[0], Sequence) and not isinstance(raw_scores[0], (str, bytes)):
        raw_scores = raw_scores[0]
    return [item for item in raw_scores if isinstance(item, Mapping)]


def _emotion_distribution(raw_scores: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Map the six-label emotion pipeline into N/N/P without guessing labels."""

    buckets = np.zeros(3, dtype=np.float64)
    unknown_labels: list[str] = []
    for item in _flatten_classifier_scores(raw_scores):
        label = str(item.get("label", "")).strip().lower()
        score = _safe_float(item.get("score"), 0.0) or 0.0
        if score <= 0.0:
            continue
        if label in {"sadness", "fear", "anger", "disgust", "negative"}:
            buckets[0] += score
        elif label in {"surprise", "neutral"}:
            buckets[1] += score
        elif label in {"joy", "love", "happiness", "positive"}:
            buckets[2] += score
        else:
            unknown_labels.append(label or "<missing>")

    total = float(buckets.sum())
    evidence = {
        "emotion_bucket_scores": {
            CLASS_NAMES[index]: float(buckets[index]) for index in range(3)
        },
        "unknown_emotion_labels": unknown_labels,
    }
    if total <= 0.0:
        return None, evidence
    return _normalise_probs(buckets), evidence


@_signal_result("Speech", 0.10)
def run_speech_pillar(
    wav_path: str | Path | None,
    whisper_model: Any,
    emotion_classifier: Callable[[str], Any],
    *,
    spellchecker: Any | None = None,
    audio_loader: Callable[..., tuple[np.ndarray, int]] | None = None,
    sample_rate: int = 16_000,
    min_duration_seconds: float = 0.5,
    min_words: int = 4,
    min_lexical_validity: float = 0.70,
) -> AudioPillarResult:
    """Return a quality-gated transcript-emotion result.

    The injected ``whisper_model`` should expose ``transcribe(path, fp16=False)``
    and return the normal Whisper mapping with ``text`` and ``segments``.  The
    injected ``emotion_classifier`` is a Hugging Face text-classification
    callable.  Neither model is loaded by this function.
    """

    loaded = _load_mono_audio(wav_path, sample_rate=sample_rate, audio_loader=audio_loader)
    if isinstance(loaded, AudioPillarResult):
        return loaded
    audio, rate = loaded
    duration = audio.size / rate
    if duration < min_duration_seconds:
        return _abstain("audio_too_short", evidence={"duration_seconds": duration})

    _rms, active_fraction, rms_mean, _rms_variation = _rms_and_activity(audio)
    if rms_mean < 3e-4 or active_fraction < 0.08:
        return _abstain(
            "insufficient_active_audio",
            evidence={
                "duration_seconds": duration,
                "active_fraction": active_fraction,
                "rms_mean": rms_mean,
            },
        )

    try:
        whisper_result = _call_transcriber(whisper_model, str(wav_path))
    except Exception as exc:
        return _abstain("transcription_failed", status="error", evidence={"error": str(exc)})

    transcript = str(whisper_result.get("text", "")).strip()
    if not transcript:
        return _abstain("no_transcript", evidence={"duration_seconds": duration})

    segment_evidence, segment_problem = _segment_metrics(whisper_result.get("segments"))
    evidence: dict[str, Any] = {
        "transcript": transcript,
        "duration_seconds": duration,
        "active_fraction": active_fraction,
        "rms_mean": rms_mean,
        **segment_evidence,
    }
    if segment_problem:
        return _abstain(segment_problem, evidence=evidence)

    no_speech = float(segment_evidence["no_speech_prob"])
    logprob = float(segment_evidence["avg_logprob"])
    compression = float(segment_evidence["compression_ratio"])
    if no_speech > _MAX_NO_SPEECH_PROB:
        return _abstain("whisper_no_speech", evidence=evidence)
    if logprob < _MIN_AVG_LOGPROB:
        return _abstain("whisper_low_logprob", evidence=evidence)
    if compression > _MAX_COMPRESSION_RATIO:
        return _abstain("whisper_repetitive_transcript", evidence=evidence)

    words = _speech_words(transcript)
    evidence["word_count"] = len(words)
    if len(words) < min_words:
        return _abstain("too_few_words", evidence=evidence)

    checker = _get_spellchecker(spellchecker)
    if checker is None:
        return _abstain("lexical_guard_unavailable", status="error", evidence=evidence)
    lexical_validity = _lexical_validity(words, checker)
    evidence["lexical_validity"] = lexical_validity
    if lexical_validity < min_lexical_validity:
        return _abstain("low_lexical_validity", evidence=evidence)

    try:
        classifier_scores = emotion_classifier(transcript)
    except Exception as exc:
        return _abstain("emotion_classifier_failed", status="error", evidence={**evidence, "error": str(exc)})
    raw_probs, classifier_evidence = _emotion_distribution(classifier_scores)
    evidence.update(classifier_evidence)
    if raw_probs is None:
        return _abstain("unmapped_emotion_labels", evidence=evidence)

    concentration = _clip01((float(np.max(raw_probs)) - 1.0 / 3.0) / (2.0 / 3.0))
    evidence["emotion_concentration"] = concentration
    if concentration < 0.08:
        return _abstain("emotion_model_ambiguous", evidence=evidence)

    content_quality = _clip01(len(words) / 12.0)
    activity_quality = _clip01(active_fraction / 0.45)
    whisper_quality = float(segment_evidence["whisper_quality"])
    quality = _clip01(
        0.25 * activity_quality
        + 0.35 * whisper_quality
        + 0.25 * lexical_validity
        + 0.15 * content_quality
    )
    reliability = _clip01(quality * (0.35 + 0.65 * concentration))
    evidence.update(
        {
            "activity_quality": activity_quality,
            "content_quality": content_quality,
            "quality": quality,
        }
    )
    if reliability < 0.40:
        return _abstain("speech_evidence_too_weak", evidence=evidence)

    # This deliberately caps even pristine text at a 0.75-ish winner.  A
    # transcript classifier is one small expert, not a source of certainty.
    smoothing_strength = min(0.65, quality * (0.35 + 0.45 * concentration))
    probs = _UNIFORM + smoothing_strength * (raw_probs - _UNIFORM)
    evidence["smoothing_strength"] = smoothing_strength
    return AudioPillarResult(
        probs=probs,
        reliability=reliability,
        available=True,
        status="used",
        reason="quality_gated_transcript",
        evidence=evidence,
    )


def _pitch_metrics(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float, bool]:
    """Estimate f0/voicing; return an empty result rather than manufacture it."""

    try:
        f0, voiced_flags, _voiced_prob = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
    except Exception:
        return np.array([], dtype=np.float64), 0.0, False

    f0_array = np.asarray(f0, dtype=np.float64).reshape(-1)
    flags = np.asarray(voiced_flags, dtype=bool).reshape(-1)
    frame_count = max(f0_array.size, flags.size)
    if frame_count == 0:
        return np.array([], dtype=np.float64), 0.0, True
    if flags.size == f0_array.size:
        valid_mask = flags & np.isfinite(f0_array)
    else:
        valid_mask = np.isfinite(f0_array)
    voiced = f0_array[valid_mask]
    voiced_fraction = float(voiced.size / frame_count)
    return voiced, voiced_fraction, True


def _spectral_flux(audio: np.ndarray, hop_length: int = 512) -> float:
    try:
        spectrum = np.abs(librosa.stft(audio, hop_length=hop_length))
        if spectrum.shape[1] < 2:
            return 0.0
        change = np.diff(spectrum, axis=1)
        normaliser = float(np.mean(spectrum**2)) + 1e-8
        return _clip01(float(np.log1p(np.mean(change**2) / normaliser)) / 1.4)
    except Exception:
        return 0.0


def _beat_strength(audio: np.ndarray, sample_rate: int) -> float:
    try:
        tempo, _beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
        tempo_value = _safe_float(tempo, 0.0) or 0.0
        # Moderate regular pulse is a weak arousal cue, not a happiness label.
        return _clip01(1.0 - abs(tempo_value - 120.0) / 120.0)
    except Exception:
        return 0.0


@_signal_result("Acoustic", 0.25)
def run_acoustic_pillar(
    wav_path: str | Path | None,
    *,
    audio_loader: Callable[..., tuple[np.ndarray, int]] | None = None,
    sample_rate: int = 22_050,
    min_duration_seconds: float = 0.5,
    min_voiced_fraction: float = 0.06,
) -> AudioPillarResult:
    """Return a bounded, quality-gated prosody result.

    Librosa measurements can reveal activity and vocal-prosody structure, but
    cannot establish the semantic sentiment of a scene.  Therefore this
    function abstains for silence, ambient/no-voice sound, and likely music;
    any retained distribution is explicitly capped near uniform.
    """

    loaded = _load_mono_audio(wav_path, sample_rate=sample_rate, audio_loader=audio_loader)
    if isinstance(loaded, AudioPillarResult):
        return loaded
    audio, rate = loaded
    duration = audio.size / rate
    if duration < min_duration_seconds:
        return _abstain("audio_too_short", evidence={"duration_seconds": duration})

    _rms, active_fraction, rms_mean, rms_variation = _rms_and_activity(audio)
    evidence: dict[str, Any] = {
        "duration_seconds": duration,
        "active_fraction": active_fraction,
        "rms_mean": rms_mean,
        "rms_variation": rms_variation,
    }
    if rms_mean < 3e-4 or active_fraction < 0.08:
        return _abstain("insufficient_active_audio", evidence=evidence)

    voiced_f0, voiced_fraction, pitch_available = _pitch_metrics(audio, rate)
    pitch_mean = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_cv = float(np.std(voiced_f0) / (pitch_mean + 1e-8)) if voiced_f0.size >= 3 else 0.0

    try:
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = float(np.sum(np.square(harmonic)))
        percussive_energy = float(np.sum(np.square(percussive)))
    except Exception:
        harmonic_energy = percussive_energy = 0.0
    harmonic_fraction = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)

    try:
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    except Exception:
        flatness = 1.0
    flatness = _clip01(flatness)
    # Used only if pYIN cannot provide f0.  A harmonic/noisy distinction is a
    # weak voice proxy, and subsequent music gating stops it from elevating a
    # tonal soundtrack to apparent speech.
    harmonic_proxy = _clip01((harmonic_fraction - 0.25) / 0.65) * (1.0 - flatness)
    if pitch_available:
        voiced_strength = _clip01(0.75 * (voiced_fraction / 0.45) + 0.25 * harmonic_proxy)
        voiced_source = "pyin"
    else:
        voiced_strength = _clip01(0.30 * harmonic_proxy)
        voiced_source = "harmonic_proxy"

    steady_signal = 1.0 - _clip01(rms_variation / 0.70)
    tonal_signal = _clip01((harmonic_fraction - 0.55) / 0.35) * (1.0 - flatness)
    low_voicing = 1.0 - _clip01(voiced_fraction / max(min_voiced_fraction * 2.5, 0.15))
    music_ambiguity = _clip01(tonal_signal * (0.55 + 0.45 * steady_signal) * low_voicing)
    evidence.update(
        {
            "pitch_available": pitch_available,
            "voiced_fraction": voiced_fraction,
            "voiced_strength": voiced_strength,
            "voiced_source": voiced_source,
            "pitch_mean_hz": pitch_mean,
            "pitch_cv": pitch_cv,
            "harmonic_fraction": harmonic_fraction,
            "spectral_flatness": flatness,
            "music_ambiguity": music_ambiguity,
        }
    )

    if music_ambiguity >= 0.60 and voiced_fraction < min_voiced_fraction:
        return _abstain("music_ambiguous", evidence=evidence)
    if voiced_strength < 0.20 or (pitch_available and voiced_fraction < min_voiced_fraction):
        return _abstain("no_reliable_voiced_signal", evidence=evidence)

    flux = _spectral_flux(audio)
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=rate)))
    except Exception:
        centroid = 0.0
    brightness = _clip01(centroid / 4_000.0)
    try:
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_variation = float(np.std(zcr) / (np.mean(zcr) + 1e-8))
    except Exception:
        zcr_variation = 0.0
    rhythm = _beat_strength(audio, rate)

    pitch_variation = _clip01(pitch_cv / 0.55)
    energy_variation = _clip01(rms_variation / 0.80)
    zcr_variation = _clip01(zcr_variation / 1.25)
    distress_cue = _clip01(
        0.34 * pitch_variation
        + 0.27 * energy_variation
        + 0.24 * flux
        + 0.15 * zcr_variation
    )
    uplift_cue = _clip01(0.40 * rhythm + 0.35 * brightness + 0.25 * _clip01(rms_mean / 0.10))
    evidence.update(
        {
            "spectral_flux": flux,
            "spectral_centroid_hz": centroid,
            "rhythm_strength": rhythm,
            "distress_cue": distress_cue,
            "uplift_cue": uplift_cue,
        }
    )

    # Cues are deliberately weak relative to a neutral prior.  They describe
    # sound dynamics, not semantic mood, and will be additionally blended below.
    raw = _normalise_probs(
        [
            0.30 + 0.70 * distress_cue,
            0.85 - 0.25 * max(distress_cue, uplift_cue),
            0.30 + 0.70 * uplift_cue,
        ]
    )
    duration_quality = _clip01(duration / 2.0)
    activity_quality = _clip01(active_fraction / 0.45)
    reliability = _clip01(
        duration_quality
        * (0.30 * activity_quality + 0.50 * voiced_strength + 0.20 * (1.0 - music_ambiguity))
    )
    evidence["quality"] = reliability
    if reliability < 0.35:
        return _abstain("acoustic_evidence_too_weak", evidence=evidence)

    # At most 42% of the distance from uniform is retained.  Even a maximal
    # acoustic cue therefore cannot output a fake 100% Negative/Positive vote.
    smoothing_strength = min(0.42, 0.14 + 0.35 * reliability)
    probs = _UNIFORM + smoothing_strength * (raw - _UNIFORM)
    evidence["smoothing_strength"] = smoothing_strength
    return AudioPillarResult(
        probs=probs,
        reliability=reliability,
        available=True,
        status="used",
        reason="bounded_voiced_acoustic_cues",
        evidence=evidence,
    )


__all__ = [
    "AudioPillarResult",
    "CLASS_NAMES",
    "run_speech_pillar",
    "run_acoustic_pillar",
]
