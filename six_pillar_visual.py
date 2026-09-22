"""Conservative visual pillars for the six-pillar SceneMotion app.

This module deliberately separates *measurement quality* from a sentiment
vote.  CLIP remains the primary spatial signal, but a near-uniform CLIP output
is reported as ``visual_uncertain`` rather than silently receiving the same
authority as a clear visual result.  The colour pillar is intentionally weak:
lighting and palette are context, not a reliable emotion detector.

The module has no Streamlit dependency so it can be tested independently and
used by both the UI and offline evaluation code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


CLASS_NAMES = ("Negative", "Neutral", "Positive")
UNIFORM = np.full(3, 1.0 / 3.0, dtype=np.float32)

# Equal-sized prompt groups avoid giving a class an advantage merely because it
# has more chances to win a max-similarity comparison.
DEFAULT_TEXT_PROMPTS: Mapping[str, tuple[str, ...]] = {
    "Negative": (
        "a video scene conveying sadness",
        "a video scene conveying fear",
        "a video scene conveying anger",
        "a video scene conveying disgust",
    ),
    "Neutral": (
        "a video scene conveying calm",
        "a neutral everyday scene",
        "an ordinary conversational scene",
        "a video scene with no clear emotion",
    ),
    "Positive": (
        "a video scene conveying joy",
        "a video scene conveying love",
        "a video scene conveying happiness",
        "a video scene conveying celebration",
    ),
}

# These are eligibility safeguards, not calibrated probabilities.  They keep a
# 34/33/33 technical argmax from being treated as a decisive spatial anchor.
SPATIAL_MIN_MARGIN = 0.05
SPATIAL_MIN_FRAME_AGREEMENT = 0.50


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return (exponentials / float(exponentials.sum())).astype(np.float32)


def _normalized(values: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("A pillar probability distribution must contain three finite non-negative values.")
    total = float(result.sum())
    if total <= 0:
        raise ValueError("A pillar probability distribution must have positive mass.")
    return result / total


def _pillar_result(
    *,
    name: str,
    probs: np.ndarray,
    reliability: float,
    available: bool,
    contributes_to_valence: bool,
    status: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "probs": _normalized(probs),
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
        "available": bool(available),
        "contributes_to_valence": bool(contributes_to_valence),
        "status": status,
        "evidence": dict(evidence),
    }


def spatial_result_from_frame_scores(frame_scores: np.ndarray) -> dict[str, Any]:
    """Turn ``[frames, 3]`` CLIP group similarities into a spatial result.

    Scores are converted with the legacy app's modest ``x10`` scale so this is
    not an unvalidated confidence boost.  The result also records temporal
    agreement between sampled frames, which the old UI discarded.
    """

    scores = np.asarray(frame_scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != 3 or scores.shape[0] == 0:
        return _pillar_result(
            name="Spatial",
            probs=UNIFORM,
            reliability=0.0,
            available=False,
            contributes_to_valence=False,
            status="no_visual_frames",
            evidence={"frames": 0},
        )

    mean_scores = scores.mean(axis=0)
    probs = _softmax((mean_scores - float(mean_scores.max())) * 10.0)
    top_index = int(np.argmax(probs))
    sorted_probs = np.sort(probs)
    margin = float(sorted_probs[-1] - sorted_probs[-2])

    frame_probs = np.stack([_softmax((row - float(row.max())) * 10.0) for row in scores])
    frame_labels = np.argmax(frame_probs, axis=1)
    agreement = float(np.mean(frame_labels == top_index))
    # A quality value for fusion, not a claim of model accuracy.
    reliability = float(np.clip(0.55 * (margin / 0.15) + 0.45 * agreement, 0.0, 1.0))
    anchor_eligible = margin >= SPATIAL_MIN_MARGIN and agreement >= SPATIAL_MIN_FRAME_AGREEMENT
    return _pillar_result(
        name="Spatial",
        probs=probs,
        reliability=reliability,
        available=True,
        contributes_to_valence=True,
        status="spatial_anchor" if anchor_eligible else "visual_uncertain",
        evidence={
            "frames": int(scores.shape[0]),
            "top_label": CLASS_NAMES[top_index],
            "top_two_margin": margin,
            "frame_agreement": agreement,
            "anchor_eligible": anchor_eligible,
            "mean_similarity": {name: float(score) for name, score in zip(CLASS_NAMES, mean_scores, strict=True)},
        },
    )


@torch.inference_mode()
def run_spatial_pillar(
    frames: Sequence[np.ndarray],
    *,
    clip_model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: str | torch.device,
    text_prompts: Mapping[str, Sequence[str]] = DEFAULT_TEXT_PROMPTS,
) -> dict[str, Any]:
    """Run the existing OpenCLIP model and return an explainable spatial result."""

    if not frames:
        return spatial_result_from_frame_scores(np.empty((0, 3), dtype=np.float32))

    groups = tuple(tuple(text_prompts[name]) for name in CLASS_NAMES)
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Spatial prompt groups must have equal sizes.")

    images = torch.stack([preprocess(Image.fromarray(np.asarray(frame, dtype=np.uint8))) for frame in frames]).to(device)
    flat_prompts = [prompt for group in groups for prompt in group]
    tokens = tokenizer(flat_prompts).to(device)
    image_features = clip_model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    text_features = clip_model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarities = (image_features @ text_features.T).detach().float().cpu().numpy()
    group_size = len(groups[0])
    # Mean aggregation makes every prompt an equal vote instead of selecting a
    # convenient best prompt independently for each emotion class.
    frame_scores = np.stack(
        [similarities[:, start : start + group_size].mean(axis=1) for start in range(0, len(flat_prompts), group_size)],
        axis=1,
    )
    return spatial_result_from_frame_scores(frame_scores)


def _crop_content(frame: np.ndarray) -> np.ndarray:
    """Remove a small border so letterbox edges do not dominate colour stats."""

    image = np.asarray(frame, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Colour frames must be RGB images with three channels.")
    height, width = image.shape[:2]
    y_pad = max(1, int(height * 0.04))
    x_pad = max(1, int(width * 0.03))
    if height <= 2 * y_pad or width <= 2 * x_pad:
        return image
    return image[y_pad:-y_pad, x_pad:-x_pad]


def run_color_pillar(frames: Sequence[np.ndarray]) -> dict[str, Any]:
    """Return conservative lighting/palette context without color stereotypes.

    Colour only moves a distribution a small distance from uniform.  It cannot
    make a clip look strongly positive or negative by itself.
    """

    if not frames:
        return _pillar_result(
            name="Color",
            probs=UNIFORM,
            reliability=0.0,
            available=False,
            contributes_to_valence=False,
            status="no_visual_frames",
            evidence={},
        )

    brightness: list[float] = []
    saturation: list[float] = []
    hue_histogram = np.zeros(180, dtype=np.float64)
    for frame in frames:
        hsv = cv2.cvtColor(_crop_content(frame), cv2.COLOR_RGB2HSV)
        hue, sat, value = cv2.split(hsv)
        brightness.append(float(value.mean() / 255.0))
        saturation.append(float(sat.mean() / 255.0))
        valid = (sat > 35) & (value > 25)
        if np.any(valid):
            hue_histogram += np.bincount(hue[valid].ravel(), minlength=180)

    mean_brightness = float(np.mean(brightness))
    mean_saturation = float(np.mean(saturation))
    contrast = float(np.std(brightness))
    colorful_fraction = float(hue_histogram.sum() / max(1, len(frames)))
    if mean_saturation < 0.06 or colorful_fraction < 20:
        return _pillar_result(
            name="Color",
            probs=UNIFORM,
            reliability=0.0,
            available=True,
            contributes_to_valence=False,
            status="low_color_information",
            evidence={"mean_brightness": mean_brightness, "mean_saturation": mean_saturation},
        )

    # Lighting is a weak contextual cue.  Its maximum effect is deliberately
    # ±0.08 before normalization; hue is reported as evidence only.
    light_shift = float(np.clip(mean_brightness - 0.5, -0.5, 0.5)) * 0.16
    probs = UNIFORM.astype(np.float64).copy()
    probs[0] += max(-light_shift, 0.0)
    probs[2] += max(light_shift, 0.0)
    # Preserve a neutral option when lighting is close to midrange.
    probs[1] += max(0.0, 0.04 - abs(light_shift))
    probs = _normalized(probs)
    reliability = float(np.clip(mean_saturation * 1.5 + contrast * 2.0, 0.0, 0.35))
    return _pillar_result(
        name="Color",
        probs=probs,
        reliability=reliability,
        available=True,
        contributes_to_valence=True,
        status="weak_lighting_context",
        evidence={
            "mean_brightness": mean_brightness,
            "mean_saturation": mean_saturation,
            "brightness_variation": contrast,
            "note": "Palette and lighting are bounded context, not a direct emotion inference.",
        },
    )


__all__ = [
    "CLASS_NAMES",
    "DEFAULT_TEXT_PROMPTS",
    "SPATIAL_MIN_FRAME_AGREEMENT",
    "SPATIAL_MIN_MARGIN",
    "run_color_pillar",
    "run_spatial_pillar",
    "spatial_result_from_frame_scores",
]
