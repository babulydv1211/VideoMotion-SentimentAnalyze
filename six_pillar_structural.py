"""Conservative structural measurements for the Motion and Temporal pillars.

This module deliberately separates *measurement* from *valence*.  Optical
flow, cuts, and pacing describe how a clip moves or is edited, but they do not
by themselves establish that a scene is positive, neutral, or negative.  Both
public pillar functions therefore return a uniform valence distribution and
``contributes_to_valence=False``.  A caller may display the evidence or use
the reliability later in a validated fusion model without turning filmmaking
style into a sentiment rule.

Frames are expected in OpenCV/BGR order.  Only ``cv2`` and ``numpy`` are used
for video analysis so the module stays usable by the local Streamlit app.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import cv2
import numpy as np


CLASS_ORDER = ("negative", "neutral", "positive")
_EPSILON = 1e-8


def uniform_valence_probs() -> np.ndarray:
    """Return a new explicit abstention distribution in ``CLASS_ORDER`` order."""

    return np.full(3, 1.0 / 3.0, dtype=np.float32)


def _base_result(
    name: str,
    status: str,
    *,
    available: bool,
    reliability: float,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build the shared public result contract for a structural pillar."""

    return {
        "name": name,
        "probs": uniform_valence_probs(),
        "class_order": CLASS_ORDER,
        "available": bool(available),
        "reliability": float(np.clip(reliability, 0.0, 1.0)),
        "status": status,
        "contributes_to_valence": False,
        "abstention_reason": "structural_measurements_are_not_validated_valence_evidence",
        "evidence": evidence,
    }


def _frame_list(frames: Sequence[np.ndarray] | np.ndarray | Iterable[np.ndarray] | None) -> list[np.ndarray]:
    """Normalize a frame container while treating a single image as one frame."""

    if frames is None:
        return []
    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:
            return [frame for frame in frames]
        if frames.ndim in (2, 3):
            return [frames]
        return []
    try:
        return list(frames)
    except TypeError:
        return []


def _to_gray(frame: np.ndarray, *, max_dimension: int) -> np.ndarray | None:
    """Convert one BGR/BGRA/gray frame to a bounded analysis-sized gray image."""

    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        return None

    if not np.isfinite(frame).all():
        return None

    image = frame
    if image.dtype != np.uint8:
        image = image.astype(np.float32, copy=False)
        # Float frames from common callers are either [0, 1] or [0, 255].
        if image.size and float(np.max(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    try:
        if image.ndim == 2:
            gray = image
        elif image.shape[2] == 1:
            gray = image[:, :, 0]
        elif image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            return None
    except cv2.error:
        return None

    height, width = gray.shape[:2]
    if height < 2 or width < 2:
        return None

    longest_side = max(height, width)
    if longest_side > max_dimension:
        scale = float(max_dimension) / float(longest_side)
        target_width = max(2, int(round(width * scale)))
        target_height = max(2, int(round(height * scale)))
        gray = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(gray)


def _prepare_gray_frames(
    frames: Sequence[np.ndarray] | np.ndarray | Iterable[np.ndarray] | None,
    *,
    max_dimension: int,
) -> tuple[list[np.ndarray], int]:
    """Return valid grayscale frames at one common shape and the input count."""

    source_frames = _frame_list(frames)
    prepared: list[np.ndarray] = []
    target_shape: tuple[int, int] | None = None

    for source in source_frames:
        gray = _to_gray(source, max_dimension=max_dimension)
        if gray is None:
            continue
        if target_shape is None:
            target_shape = gray.shape
        elif gray.shape != target_shape:
            gray = cv2.resize(gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        prepared.append(gray)

    return prepared, len(source_frames)


def _sample_pair_indices(frame_count: int, max_pairs: int) -> np.ndarray:
    """Evenly sample adjacent frame-pair indices without altering their order."""

    pair_count = max(0, frame_count - 1)
    if pair_count == 0:
        return np.empty(0, dtype=np.int32)
    if max_pairs <= 0 or pair_count <= max_pairs:
        return np.arange(pair_count, dtype=np.int32)
    return np.unique(np.linspace(0, pair_count - 1, num=max_pairs, dtype=np.int32))


def _texture_quality(gray: np.ndarray) -> float:
    """Estimate whether a frame has enough contrast/edges for structural cues."""

    # A stationary but textured scene is still a high-quality *measurement*.
    contrast = min(1.0, float(np.std(gray)) / 48.0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = float(np.mean(np.abs(grad_x) + np.abs(grad_y))) / 255.0
    edge_quality = min(1.0, edge_strength / 0.15)
    return float(np.clip(0.55 * contrast + 0.45 * edge_quality, 0.0, 1.0))


def _flow_direction_entropy(flow: np.ndarray, magnitude: np.ndarray) -> float:
    """Return normalized direction entropy for active flow vectors, in [0, 1]."""

    active = magnitude > 0.5
    if not np.any(active):
        return 0.0
    angles = np.arctan2(flow[..., 1][active], flow[..., 0][active])
    histogram, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    probabilities = histogram.astype(np.float64) / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0.0]
    if probabilities.size <= 1:
        return 0.0
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.clip(entropy / np.log(8.0), 0.0, 1.0))


def measure_motion(
    frames: Sequence[np.ndarray] | np.ndarray | Iterable[np.ndarray] | None,
    *,
    max_pairs: int = 48,
    max_dimension: int = 320,
) -> dict[str, Any]:
    """Measure dense optical flow and distinguish global from local movement.

    ``global_motion_px`` is a robust median camera/whole-scene translation
    proxy.  ``local_motion_px`` is the residual motion after that translation,
    so it is useful for describing movement inside an otherwise stable shot.
    Their values are analysis-frame pixels per sampled adjacent frame pair.

    The returned probability vector is always uniform.  ``reliability`` means
    *optical-flow measurement quality* (texture plus sequence coverage), not
    confidence in any sentiment label.
    """

    gray_frames, input_count = _prepare_gray_frames(frames, max_dimension=max_dimension)
    evidence: dict[str, Any] = {
        "input_frame_count": int(input_count),
        "valid_frame_count": int(len(gray_frames)),
        "pairs_examined": 0,
    }
    if len(gray_frames) < 2:
        evidence["reason"] = "at_least_two_valid_frames_are_required_for_optical_flow"
        return _base_result("motion", "insufficient_frames", available=False, reliability=0.0, evidence=evidence)

    pair_indices = _sample_pair_indices(len(gray_frames), max_pairs)
    global_magnitudes: list[float] = []
    local_magnitudes: list[float] = []
    mean_magnitudes: list[float] = []
    active_fractions: list[float] = []
    coherences: list[float] = []
    direction_entropies: list[float] = []
    pair_qualities: list[float] = []

    for pair_index in pair_indices.tolist():
        previous = gray_frames[pair_index]
        current = gray_frames[pair_index + 1]
        try:
            flow = cv2.calcOpticalFlowFarneback(
                previous,
                current,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        except cv2.error:
            continue

        if flow is None or not np.isfinite(flow).all():
            continue
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        global_vector = np.median(flow.reshape(-1, 2), axis=0)
        residual = flow - global_vector.reshape(1, 1, 2)
        residual_magnitude = cv2.magnitude(residual[..., 0], residual[..., 1])

        mean_magnitude = float(np.mean(magnitude))
        global_magnitude = float(np.linalg.norm(global_vector))
        local_magnitude = float(np.mean(residual_magnitude))
        global_magnitudes.append(global_magnitude)
        local_magnitudes.append(local_magnitude)
        mean_magnitudes.append(mean_magnitude)
        active_fractions.append(float(np.mean(magnitude > 0.5)))
        coherences.append(float(np.clip(global_magnitude / (mean_magnitude + _EPSILON), 0.0, 1.0)))
        direction_entropies.append(_flow_direction_entropy(flow, magnitude))
        pair_qualities.append(0.5 * (_texture_quality(previous) + _texture_quality(current)))

    evidence["pairs_examined"] = int(len(pair_indices))
    evidence["valid_pairs"] = int(len(mean_magnitudes))
    if not mean_magnitudes:
        evidence["reason"] = "optical_flow_could_not_be_estimated"
        return _base_result("motion", "no_valid_frame_pairs", available=False, reliability=0.0, evidence=evidence)

    mean_global = float(np.mean(global_magnitudes))
    mean_local = float(np.mean(local_magnitudes))
    mean_motion = float(np.mean(mean_magnitudes))
    global_fraction = float(np.clip(mean_global / (mean_global + mean_local + _EPSILON), 0.0, 1.0))
    sequence_coverage = min(1.0, len(mean_magnitudes) / 8.0)
    measurement_quality = float(np.mean(pair_qualities))
    reliability = 0.55 * measurement_quality + 0.45 * sequence_coverage

    analysis_height, analysis_width = gray_frames[0].shape
    evidence.update(
        {
            "analysis_width": int(analysis_width),
            "analysis_height": int(analysis_height),
            "mean_motion_px": mean_motion,
            "median_motion_px": float(np.median(mean_magnitudes)),
            "global_motion_px": mean_global,
            "local_motion_px": mean_local,
            "global_motion_fraction": global_fraction,
            "localized_motion_fraction": float(1.0 - global_fraction),
            "global_flow_coherence": float(np.mean(coherences)),
            "active_motion_fraction": float(np.mean(active_fractions)),
            "direction_entropy": float(np.mean(direction_entropies)),
            "mean_texture_quality": measurement_quality,
            "sequence_coverage": sequence_coverage,
            "motion_interpretation": "descriptive_only_not_a_valence_vote",
        }
    )
    return _base_result("motion", "measured", available=True, reliability=reliability, evidence=evidence)


def _histogram_distance(previous: np.ndarray, current: np.ndarray) -> float:
    """Return a stable [0, 1] grayscale-histogram distance for cut detection."""

    hist_previous = cv2.calcHist([previous], [0], None, [32], [0, 256])
    hist_current = cv2.calcHist([current], [0], None, [32], [0, 256])
    hist_previous = cv2.normalize(hist_previous, hist_previous).reshape(-1)
    hist_current = cv2.normalize(hist_current, hist_current).reshape(-1)
    distance = cv2.compareHist(hist_previous.astype(np.float32), hist_current.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return float(np.clip(distance, 0.0, 1.0))


def measure_temporal(
    frames: Sequence[np.ndarray] | np.ndarray | Iterable[np.ndarray] | None,
    *,
    fps: float | None = None,
    max_pairs: int = 96,
    max_dimension: int = 320,
    cut_threshold: float = 0.45,
) -> dict[str, Any]:
    """Measure frame changes, shot cuts, and pacing without assigning valence.

    A cut score combines normalized mean absolute frame difference and
    histogram change.  It is a useful editing measurement, not a claim that a
    quick-cut scene is positive or negative.  ``fps`` is optional and only
    enables time-normalized rates in the evidence.
    """

    gray_frames, input_count = _prepare_gray_frames(frames, max_dimension=max_dimension)
    evidence: dict[str, Any] = {
        "input_frame_count": int(input_count),
        "valid_frame_count": int(len(gray_frames)),
        "pairs_examined": 0,
        "cut_threshold": float(cut_threshold),
    }
    if len(gray_frames) < 2:
        evidence["reason"] = "at_least_two_valid_frames_are_required_for_temporal_measurement"
        return _base_result("temporal", "insufficient_frames", available=False, reliability=0.0, evidence=evidence)

    pair_indices = _sample_pair_indices(len(gray_frames), max_pairs)
    differences: list[float] = []
    histogram_distances: list[float] = []
    cut_scores: list[float] = []
    transition_qualities: list[float] = []

    for pair_index in pair_indices.tolist():
        previous = gray_frames[pair_index]
        current = gray_frames[pair_index + 1]
        difference = float(np.mean(cv2.absdiff(previous, current))) / 255.0
        histogram_distance = _histogram_distance(previous, current)
        cut_score = float(np.clip(0.65 * difference + 0.35 * histogram_distance, 0.0, 1.0))
        differences.append(difference)
        histogram_distances.append(histogram_distance)
        cut_scores.append(cut_score)
        transition_qualities.append(0.5 * (_texture_quality(previous) + _texture_quality(current)))

    evidence["pairs_examined"] = int(len(pair_indices))
    evidence["valid_pairs"] = int(len(differences))
    if not differences:
        evidence["reason"] = "frame_changes_could_not_be_estimated"
        return _base_result("temporal", "no_valid_frame_pairs", available=False, reliability=0.0, evidence=evidence)

    difference_array = np.asarray(differences, dtype=np.float32)
    cut_score_array = np.asarray(cut_scores, dtype=np.float32)
    cut_count = int(np.count_nonzero(cut_score_array >= cut_threshold))
    cut_rate = float(cut_count / len(cut_score_array))
    sequence_coverage = min(1.0, len(difference_array) / 8.0)
    # Cuts and mean visual activity are deliberately retained as separate
    # descriptive measures.  "Pacing" here is edit/activity density only.
    pacing_index = float(np.clip(0.60 * float(np.mean(difference_array)) + 0.40 * cut_rate, 0.0, 1.0))
    reliability = 0.35 * float(np.mean(transition_qualities)) + 0.40 * sequence_coverage + 0.25

    evidence.update(
        {
            "frame_difference_mean": float(np.mean(difference_array)),
            "frame_difference_median": float(np.median(difference_array)),
            "frame_difference_p90": float(np.percentile(difference_array, 90)),
            "frame_difference_std": float(np.std(difference_array)),
            "histogram_distance_mean": float(np.mean(histogram_distances)),
            "cut_score_mean": float(np.mean(cut_score_array)),
            "cut_count": cut_count,
            "cut_rate_per_sampled_transition": cut_rate,
            "mean_shot_length_frames": float(len(gray_frames) / (cut_count + 1)),
            "pacing_index": pacing_index,
            "pacing_variability": float(np.std(difference_array)),
            "mean_transition_texture_quality": float(np.mean(transition_qualities)),
            "sequence_coverage": sequence_coverage,
            "temporal_interpretation": "descriptive_only_not_a_valence_vote",
        }
    )
    if fps is not None and np.isfinite(fps) and fps > 0.0:
        duration_seconds = max(0.0, (len(gray_frames) - 1) / float(fps))
        evidence["fps"] = float(fps)
        evidence["duration_seconds"] = duration_seconds
        evidence["cut_rate_per_second"] = float(cut_count / duration_seconds) if duration_seconds > 0.0 else 0.0

    return _base_result("temporal", "measured", available=True, reliability=reliability, evidence=evidence)


def measure_structural_pillars(
    frames: Sequence[np.ndarray] | np.ndarray | Iterable[np.ndarray] | None,
    *,
    fps: float | None = None,
    max_dimension: int = 320,
) -> dict[str, dict[str, Any]]:
    """Convenience wrapper returning both non-valence structural pillars."""

    return {
        "motion": measure_motion(frames, max_dimension=max_dimension),
        "temporal": measure_temporal(frames, fps=fps, max_dimension=max_dimension),
    }


# Readable aliases for callers that prefer pillar terminology.
motion_pillar = measure_motion
temporal_pillar = measure_temporal


__all__ = [
    "CLASS_ORDER",
    "measure_motion",
    "measure_structural_pillars",
    "measure_temporal",
    "motion_pillar",
    "temporal_pillar",
    "uniform_valence_probs",
]
