"""One coherent local workflow for SceneMotion's six pillars.

The old Streamlit app called six unrelated functions that returned only raw
three-number arrays.  This module keeps the same local ingredients but makes
each output explicit about quality, availability, and whether it is valid
valence evidence before passing it to the spatial-anchor fusion policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from six_pillar_audio import AudioPillarResult, run_acoustic_pillar, run_speech_pillar
from six_pillar_fusion import FusionResult, fuse_six_pillars
from six_pillar_structural import measure_structural_pillars
from six_pillar_visual import run_color_pillar
from trained_spatial_pillar import TrainedSpatialPillar


def _audio_mapping(name: str, result: AudioPillarResult) -> dict[str, Any]:
    """Adapt the quality-gated audio result to the common six-pillar schema."""

    return {
        "name": name,
        "probs": result.probs.copy(),
        "class_order": ("Negative", "Neutral", "Positive"),
        "reliability": result.reliability,
        "available": result.available,
        "contributes_to_valence": True,
        "status": result.status,
        "evidence": {"reason": result.reason, **dict(result.evidence)},
    }


def _structural_frames_from_rgb(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    """The uploader supplies RGB frames; OpenCV structural code expects BGR."""

    converted: list[np.ndarray] = []
    for frame in frames:
        image = np.asarray(frame, dtype=np.uint8)
        if image.ndim == 3 and image.shape[2] == 3:
            converted.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            converted.append(image)
    return converted


def analyze_six_pillars(
    *,
    video_path: str | Path,
    rgb_frames: Sequence[np.ndarray],
    wav_path: str | Path | None,
    trained_spatial: TrainedSpatialPillar,
    whisper_model: Any,
    emotion_classifier: Callable[[str], Any],
) -> dict[str, Any]:
    """Run all six local pillars and return a transparent fusion record.

    The trained spatial adapter owns Pillar 1.  It runs the checkpoint's
    spatial auxiliary head only; no final prediction from the old five-pillar
    checkpoint is folded into this six-pillar result.
    """

    spatial = trained_spatial.analyze(video_path)
    speech = _audio_mapping("Speech", run_speech_pillar(wav_path, whisper_model, emotion_classifier))
    acoustic = _audio_mapping("Acoustic", run_acoustic_pillar(wav_path))
    color = run_color_pillar(rgb_frames)
    structural = measure_structural_pillars(_structural_frames_from_rgb(rgb_frames))
    # Normalize display names while retaining canonical roles in fusion.
    structural["motion"]["name"] = "Motion"
    structural["temporal"]["name"] = "Temporal"
    fusion: FusionResult = fuse_six_pillars(
        [spatial, speech, acoustic, color, structural["motion"], structural["temporal"]]
    )
    pillars = {
        "Spatial": spatial,
        "Speech": speech,
        "Acoustic": acoustic,
        "Color": color,
        "Motion": structural["motion"],
        "Temporal": structural["temporal"],
    }
    transcript = str(speech["evidence"].get("transcript", "")).strip()
    return {
        "pillars": pillars,
        "fusion": fusion,
        "transcript": transcript,
        "speech_reason": str(speech["evidence"].get("reason", "")),
    }


__all__ = ["analyze_six_pillars"]
