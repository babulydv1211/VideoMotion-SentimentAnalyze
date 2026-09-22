"""Explainable, spatial-anchor fusion for SceneMotion's six pillars.

The module deliberately keeps *evidence* separate from a final decision.  The
spatial pillar is the primary semantic signal; speech, acoustic, and colour
can provide bounded support or flag a conflict, but cannot silently replace a
clear spatial decision.  Motion and temporal measurements are retained in the
result for explanation, but never converted into a sentiment vote here.

All distributions use the fixed class order ``Negative, Neutral, Positive``.
They are relative scores, not calibrated probabilities or an accuracy claim.
The module has no Streamlit or model dependency so it can be used by the UI,
the trained spatial-head wrapper, and tests without circular imports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


CLASS_NAMES: tuple[str, str, str] = ("Negative", "Neutral", "Positive")
"""The only accepted / emitted class order for a pillar distribution."""

UNIFORM = np.full(3, 1.0 / 3.0, dtype=np.float64)

# These are contribution ceilings, not estimates of accuracy.  The historical
# root app used a 40/10/25/5/15/5 split.  Here the spatial pillar is explicitly
# the anchor: even when every supporting valence signal is usable, it keeps at
# least 70% of the decision weight.  Acoustic is capped below its old 25%
# because hand-authored prosody rules are especially prone to false certainty.
DEFAULT_BASE_WEIGHTS: dict[str, float] = {
    "spatial": 0.40,
    "speech": 0.10,
    "acoustic": 0.25,
    "color": 0.05,
    "motion": 0.15,
    "temporal": 0.05,
}


def _canonical_name(name: str) -> str:
    """Map common display names to stable internal pillar names."""

    lowered = " ".join(str(name).replace("/", " ").replace("_", " ").split()).lower()
    if "spatial" in lowered or "clip" in lowered or "visual" in lowered:
        return "spatial"
    if "speech" in lowered or "transcript" in lowered or "whisper" in lowered:
        return "speech"
    if "acoustic" in lowered or "prosody" in lowered or "audio" in lowered:
        return "acoustic"
    if "color" in lowered or "colour" in lowered or "hsv" in lowered:
        return "color"
    if "motion" in lowered or "flow" in lowered:
        return "motion"
    if "temporal" in lowered or "shot" in lowered or "cut" in lowered:
        return "temporal"
    return lowered


def _normalise_distribution(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate and normalise one Negative/Neutral/Positive distribution."""

    probs = np.asarray(values, dtype=np.float64)
    if probs.shape != (3,):
        raise ValueError("A pillar distribution must contain exactly three values: Negative, Neutral, Positive.")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("A pillar distribution must contain finite, non-negative values.")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("A pillar distribution must have positive mass.")
    return probs / total


def _clip_unit(value: Any, *, default: float) -> float:
    """Return a finite [0, 1] quality value without letting bad metadata crash UI."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    if not np.isfinite(numeric):
        numeric = default
    return float(np.clip(numeric, 0.0, 1.0))


def _as_bool(value: Any, *, default: bool) -> bool:
    """Parse ordinary boolean-like metadata without treating non-empty text as true."""

    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "available", "usable"}:
            return True
        if lowered in {"false", "no", "0", "unavailable", "abstain", "abstained"}:
            return False
        return default
    return bool(value)


def _distribution_from_mapping(mapping: Mapping[str, Any]) -> np.ndarray:
    """Read several lightweight result schemas and return the fixed class order."""

    candidate: Any | None = None
    for key in ("probs", "probabilities", "distribution", "scores", "vote"):
        if key in mapping:
            candidate = mapping[key]
            break

    if candidate is None:
        # This also permits the concise {"negative": ..., "neutral": ...,
        # "positive": ...} form in small integrations and tests.
        lowered = {str(key).lower(): value for key, value in mapping.items()}
        if all(name.lower() in lowered for name in CLASS_NAMES):
            candidate = [lowered[name.lower()] for name in CLASS_NAMES]
        elif _as_bool(mapping.get("abstain"), default=False) or not _as_bool(mapping.get("available"), default=True):
            candidate = UNIFORM
        else:
            raise ValueError("Pillar mapping needs probs/probabilities/distribution/scores in class order.")

    if isinstance(candidate, Mapping):
        lowered = {str(key).lower(): value for key, value in candidate.items()}
        try:
            candidate = [lowered[name.lower()] for name in CLASS_NAMES]
        except KeyError as error:
            raise ValueError("A score mapping must contain Negative, Neutral, and Positive values.") from error

    probs = np.asarray(candidate, dtype=np.float64)
    order = mapping.get("class_order")
    if order is not None:
        normalised_order = tuple(str(label).strip().lower() for label in order)
        expected_order = tuple(label.lower() for label in CLASS_NAMES)
        if len(normalised_order) != 3 or set(normalised_order) != set(expected_order):
            raise ValueError("class_order must contain Negative, Neutral, Positive exactly once.")
        reorder = [normalised_order.index(label) for label in expected_order]
        probs = probs[reorder]
    return _normalise_distribution(probs)


@dataclass
class PillarResult:
    """A common, dependency-free contract for any of the six pillars.

    ``reliability`` must come from a real availability/quality check owned by
    the pillar (for example, visual frame agreement or transcript quality). It
    is never inferred from how sharp a class distribution happens to be.
    """

    name: str
    probs: Sequence[float] | np.ndarray
    reliability: float = 1.0
    available: bool = True
    contributes_to_valence: bool = True
    status: str = "available"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    base_weight: float | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.probs = _normalise_distribution(self.probs)
        self.reliability = _clip_unit(self.reliability, default=0.0)
        self.available = bool(self.available)
        self.contributes_to_valence = bool(self.contributes_to_valence)
        self.status = str(self.status or "available")
        self.evidence = dict(self.evidence or {})
        if self.base_weight is not None:
            try:
                weight = float(self.base_weight)
            except (TypeError, ValueError) as error:
                raise ValueError("base_weight must be a finite non-negative number.") from error
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("base_weight must be a finite non-negative number.")
            self.base_weight = weight

    @property
    def probabilities(self) -> np.ndarray:
        """Alias for callers that use the longer output key."""

        return self.probs.copy()

    @property
    def canonical_name(self) -> str:
        return _canonical_name(self.name)

    @property
    def is_uniform(self) -> bool:
        """Uniform is the explicit abstention sentinel used by this project."""

        return bool(np.allclose(self.probs, UNIFORM, rtol=0.0, atol=1e-4))

    @property
    def top_index(self) -> int:
        return int(np.argmax(self.probs))

    @property
    def top_label(self) -> str:
        return CLASS_NAMES[self.top_index]

    @property
    def top_two_margin(self) -> float:
        ordered = np.sort(self.probs)
        return float(ordered[-1] - ordered[-2])

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PillarResult":
        """Coerce a normal pillar dictionary into the shared contract.

        The aliases deliberately cover the existing lightweight modules:
        ``probs``/``reliability``/``evidence`` and common alternatives such as
        ``probabilities``/``quality``/``metadata``.  A structural ``role`` or a
        known Motion/Temporal name is always kept out of valence fusion.
        """

        name = str(value.get("name", value.get("pillar", value.get("source", "Unknown"))))
        canonical = _canonical_name(name)
        role = str(value.get("role", "")).strip().lower()
        default_contribution = canonical not in {"motion", "temporal"} and role not in {
            "structural",
            "context",
            "non_valence",
        }
        contributes = _as_bool(
            value.get("contributes_to_valence", value.get("use_for_valence")),
            default=default_contribution,
        )
        # Motion and temporal evidence must not become a valence rule simply
        # because an upstream integration forgot this field.
        if canonical in {"motion", "temporal"}:
            contributes = False

        evidence = value.get("evidence", value.get("metadata", {}))
        if not isinstance(evidence, Mapping):
            evidence = {"raw_evidence": evidence}
        return cls(
            name=name,
            probs=_distribution_from_mapping(value),
            reliability=value.get("reliability", value.get("quality", 1.0)),
            available=_as_bool(value.get("available"), default=True),
            contributes_to_valence=contributes,
            status=str(value.get("status", "available")),
            evidence=evidence,
            base_weight=value.get("base_weight", value.get("weight")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the interoperable mapping schema without sharing mutable arrays."""

        return {
            "name": self.name,
            "probs": self.probs.copy(),
            "class_order": CLASS_NAMES,
            "reliability": self.reliability,
            "available": self.available,
            "contributes_to_valence": self.contributes_to_valence,
            "status": self.status,
            "evidence": dict(self.evidence),
            "base_weight": self.base_weight,
        }


PillarInput = PillarResult | Mapping[str, Any] | tuple[Any, float, str]


def coerce_pillar_result(value: PillarInput) -> PillarResult:
    """Accept a :class:`PillarResult`, a plain mapping, or legacy score tuple.

    The legacy ``(probs, base_weight, name)`` form is accepted only as a bridge
    for the current app while it moves to result mappings.  New callers should
    return a mapping or ``PillarResult`` so availability and evidence survive.
    """

    if isinstance(value, PillarResult):
        return value
    if isinstance(value, Mapping):
        return PillarResult.from_mapping(value)
    if isinstance(value, tuple) and len(value) == 3:
        probs, base_weight, name = value
        return PillarResult(name=str(name), probs=probs, base_weight=float(base_weight))
    raise TypeError("Each pillar must be a PillarResult, mapping, or legacy (probs, base_weight, name) tuple.")


@dataclass(frozen=True)
class FusionConfig:
    """Conservative, explainable policy knobs for spatial-anchor fusion."""

    # A supplied trained spatial head may set status="spatial_anchor" or
    # evidence["anchor_eligible"] = True.  It still needs a non-trivial lead,
    # which prevents a 34/33/33 technical maximum from becoming authoritative.
    spatial_min_top_probability: float = 0.40
    spatial_min_margin: float = 0.05
    explicit_anchor_min_margin: float = 0.02
    min_spatial_reliability: float = 0.25

    # At most 30% of a clear spatial decision can come from all non-spatial
    # valence cues together.  Their individual caps avoid missing pillars
    # redistributing their authority to one noisy source.
    max_total_support_weight: float = 0.30
    max_support_weights: Mapping[str, float] = field(
        default_factory=lambda: {"speech": 0.10, "acoustic": 0.15, "color": 0.05}
    )
    max_unknown_support_weight: float = 0.05

    # A clear anchor remains the displayed label even under severe disagreement.
    # We retain the unprotected evidence distribution so the UI can explain the
    # conflict rather than hiding it.
    protected_anchor_margin: float = 0.005
    conflict_min_top_probability: float = 0.55
    conflict_min_opposition: float = 0.20

    anchor_statuses: tuple[str, ...] = (
        "spatial_anchor",
        "trained_spatial_anchor",
        "anchor",
        "clear",
    )
    uncertain_statuses: tuple[str, ...] = (
        "visual_uncertain",
        "uncertain",
        "abstain",
        "abstained",
        "unavailable",
        "no_visual_frames",
        "error",
    )


@dataclass(frozen=True)
class PillarContribution:
    """One transparent account of how a pillar was handled by fusion."""

    name: str
    canonical_name: str
    base_weight: float
    effective_weight: float
    state: str
    used_for_valence: bool
    top_label: str
    top_score: float
    reliability: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "base_weight": self.base_weight,
            "effective_weight": self.effective_weight,
            "state": self.state,
            "used_for_valence": self.used_for_valence,
            "top_label": self.top_label,
            "top_score": self.top_score,
            "reliability": self.reliability,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ConflictFlag:
    """A high-quality supporting source strongly disagrees with spatial evidence."""

    pillar: str
    spatial_label: str
    pillar_label: str
    pillar_top_score: float
    opposition: float
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pillar": self.pillar,
            "spatial_label": self.spatial_label,
            "pillar_label": self.pillar_label,
            "pillar_top_score": self.pillar_top_score,
            "opposition": self.opposition,
            "severity": self.severity,
        }


@dataclass
class FusionResult:
    """Full fusion output, including safe display status and raw evidence."""

    probabilities: np.ndarray
    evidence_probabilities: np.ndarray
    label: str
    decision_label: str
    status: str
    spatial_label: str | None
    spatial_is_anchor: bool
    spatial_margin: float | None
    anchor_protection_applied: bool
    conflicts: tuple[ConflictFlag, ...]
    contributions: Mapping[str, PillarContribution]
    structural_evidence: Mapping[str, PillarResult]
    notes: tuple[str, ...] = ()

    @property
    def final_probs(self) -> np.ndarray:
        """Compatibility alias for callers that used the old app terminology."""

        return self.probabilities.copy()

    @property
    def scores(self) -> np.ndarray:
        """Compatibility alias; scores remain normalized relative evidence."""

        return self.probabilities.copy()

    @property
    def is_uncertain(self) -> bool:
        return self.status in {"provisional", "uncertain", "spatial_unavailable"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "probs": self.probabilities.copy(),
            "evidence_probs": self.evidence_probabilities.copy(),
            "class_order": CLASS_NAMES,
            "label": self.label,
            "decision_label": self.decision_label,
            "status": self.status,
            "spatial_label": self.spatial_label,
            "spatial_is_anchor": self.spatial_is_anchor,
            "spatial_margin": self.spatial_margin,
            "anchor_protection_applied": self.anchor_protection_applied,
            "conflicts": [flag.to_dict() for flag in self.conflicts],
            "contributions": {name: item.to_dict() for name, item in self.contributions.items()},
            "structural_evidence": {name: item.to_dict() for name, item in self.structural_evidence.items()},
            "notes": list(self.notes),
        }


def _default_base_weight(pillar: PillarResult) -> float:
    return float(DEFAULT_BASE_WEIGHTS.get(pillar.canonical_name, 0.0) if pillar.base_weight is None else pillar.base_weight)


def _is_structural(pillar: PillarResult) -> bool:
    return pillar.canonical_name in {"motion", "temporal"} or not pillar.contributes_to_valence


def _is_usable_valence(pillar: PillarResult) -> bool:
    return pillar.available and pillar.contributes_to_valence and not pillar.is_uniform and pillar.reliability > 0.0


def _spatial_is_anchor(spatial: PillarResult | None, config: FusionConfig) -> bool:
    if spatial is None or not _is_usable_valence(spatial):
        return False

    status = spatial.status.strip().lower()
    evidence_anchor = spatial.evidence.get("anchor_eligible")
    if status in {item.lower() for item in config.uncertain_statuses} or evidence_anchor is False:
        return False
    if spatial.reliability < config.min_spatial_reliability:
        return False

    explicit_anchor = status in {item.lower() for item in config.anchor_statuses} or evidence_anchor is True
    if explicit_anchor:
        return spatial.top_two_margin >= config.explicit_anchor_min_margin
    return (
        float(spatial.probs.max()) >= config.spatial_min_top_probability
        and spatial.top_two_margin >= config.spatial_min_margin
    )


def _support_weight(pillar: PillarResult, config: FusionConfig) -> float:
    """Return an absolute bounded support weight; never renormalise missing cues."""

    if not _is_usable_valence(pillar):
        return 0.0
    requested = _default_base_weight(pillar)
    allowed = float(config.max_support_weights.get(pillar.canonical_name, config.max_unknown_support_weight))
    return float(min(requested, allowed) * pillar.reliability)


def _protect_anchor_distribution(
    distribution: np.ndarray,
    anchor_index: int,
    minimum_margin: float,
) -> tuple[np.ndarray, bool]:
    """Minimally transfer mass so supporting rules cannot overturn a clear anchor."""

    protected = np.asarray(distribution, dtype=np.float64).copy()
    non_anchor = [index for index in range(3) if index != anchor_index]
    challenger = max(non_anchor, key=lambda index: float(protected[index]))
    deficit = float(protected[challenger] + minimum_margin - protected[anchor_index])
    if deficit <= 0:
        return protected, False

    # Moving x from challenger to anchor closes the gap by 2x.  The epsilon
    # cap guards a malformed distribution but should never trigger after valid
    # convex fusion.
    transfer = min(float(protected[challenger]), deficit / 2.0)
    protected[challenger] -= transfer
    protected[anchor_index] += transfer
    return _normalise_distribution(protected), True


def _conflicts_for_supports(
    spatial: PillarResult,
    supports: Iterable[PillarResult],
    config: FusionConfig,
) -> tuple[ConflictFlag, ...]:
    flags: list[ConflictFlag] = []
    spatial_index = spatial.top_index
    for pillar in supports:
        pillar_index = pillar.top_index
        opposition = float(pillar.probs[pillar_index] - pillar.probs[spatial_index])
        if (
            pillar_index != spatial_index
            and float(pillar.probs[pillar_index]) >= config.conflict_min_top_probability
            and opposition >= config.conflict_min_opposition
        ):
            severity = "strong" if opposition >= 0.50 else "moderate"
            flags.append(
                ConflictFlag(
                    pillar=pillar.name,
                    spatial_label=spatial.top_label,
                    pillar_label=pillar.top_label,
                    pillar_top_score=float(pillar.probs[pillar_index]),
                    opposition=opposition,
                    severity=severity,
                )
            )
    return tuple(flags)


def _contribution(
    pillar: PillarResult,
    *,
    effective_weight: float,
    state: str,
    used_for_valence: bool,
    reason: str,
) -> PillarContribution:
    return PillarContribution(
        name=pillar.name,
        canonical_name=pillar.canonical_name,
        base_weight=_default_base_weight(pillar),
        effective_weight=float(effective_weight),
        state=state,
        used_for_valence=used_for_valence,
        top_label=pillar.top_label,
        top_score=float(pillar.probs[pillar.top_index]),
        reliability=pillar.reliability,
        reason=reason,
    )


def fuse_six_pillars(
    pillars: Iterable[PillarInput],
    *,
    config: FusionConfig | None = None,
) -> FusionResult:
    """Fuse six result objects under an explicit spatial-anchor policy.

    A clear spatial result is the decision anchor.  Supporting valence pillars
    use their *absolute* capped weights (rather than inheriting the weights of
    abstaining signals), and their total cannot exceed 30% by default.  If they
    would reverse the anchor, the result remains anchored and records both the
    raw evidence distribution and a conflict flag.  Near-tied/missing spatial
    evidence is intentionally returned as provisional or uncertain.
    """

    policy = config or FusionConfig()
    materialised = [coerce_pillar_result(item) for item in pillars]
    if not materialised:
        raise ValueError("At least one pillar result is required.")

    spatial = next((pillar for pillar in materialised if pillar.canonical_name == "spatial"), None)
    structural = {pillar.name: pillar for pillar in materialised if _is_structural(pillar)}
    supports = [
        pillar
        for pillar in materialised
        if pillar is not spatial and not _is_structural(pillar) and _is_usable_valence(pillar)
    ]
    support_weights = {id(pillar): _support_weight(pillar, policy) for pillar in supports}
    total_support = float(sum(support_weights.values()))
    if total_support > policy.max_total_support_weight and total_support > 0:
        scale = policy.max_total_support_weight / total_support
        support_weights = {key: value * scale for key, value in support_weights.items()}
        total_support = policy.max_total_support_weight

    spatial_usable = spatial is not None and _is_usable_valence(spatial)
    spatial_anchor = _spatial_is_anchor(spatial, policy)
    contributions: dict[str, PillarContribution] = {}
    notes: list[str] = []

    # Default state for every source makes the output suitable for a UI table,
    # including abstentions and structural measurements.
    for pillar in materialised:
        if _is_structural(pillar):
            reason = "Structural evidence is shown but never used as a valence vote."
            contributions[pillar.name] = _contribution(
                pillar,
                effective_weight=0.0,
                state="structural_only",
                used_for_valence=False,
                reason=reason,
            )
        elif pillar is not spatial and not _is_usable_valence(pillar):
            reason = "Pillar abstained, was unavailable, or had zero declared reliability."
            contributions[pillar.name] = _contribution(
                pillar,
                effective_weight=0.0,
                state="abstained",
                used_for_valence=False,
                reason=reason,
            )

    if not spatial_usable:
        # We retain support evidence for inspection but refuse to present it as
        # a full substitute for the project's primary visual semantic signal.
        if total_support > 0:
            evidence = sum(
                support_weights[id(pillar)] * pillar.probs for pillar in supports
            ) / total_support
            final = _normalise_distribution(evidence)
            label = CLASS_NAMES[int(np.argmax(final))]
            status = "spatial_unavailable"
            decision_label = "Uncertain"
            notes.append("Spatial evidence was unavailable; supporting cues are diagnostic only.")
        else:
            final = UNIFORM.copy()
            evidence = final.copy()
            label = CLASS_NAMES[int(np.argmax(final))]
            status = "uncertain"
            decision_label = "Uncertain"
            notes.append("All valence pillars abstained or were unavailable.")

        for pillar in supports:
            contributions[pillar.name] = _contribution(
                pillar,
                effective_weight=support_weights[id(pillar)],
                state="diagnostic_support_only",
                used_for_valence=False,
                reason="Spatial evidence is unavailable, so this cue cannot make a final decision alone.",
            )
        if spatial is not None:
            contributions[spatial.name] = _contribution(
                spatial,
                effective_weight=0.0,
                state="abstained",
                used_for_valence=False,
                reason="Spatial pillar was unavailable, uniform, or below its declared quality threshold.",
            )
        return FusionResult(
            probabilities=final,
            evidence_probabilities=evidence,
            label=label,
            decision_label=decision_label,
            status=status,
            spatial_label=None,
            spatial_is_anchor=False,
            spatial_margin=None,
            anchor_protection_applied=False,
            conflicts=(),
            contributions=contributions,
            structural_evidence=structural,
            notes=tuple(notes),
        )

    assert spatial is not None  # narrows the type after spatial_usable
    spatial_weight = 1.0 - total_support
    evidence = spatial_weight * spatial.probs
    for pillar in supports:
        evidence += support_weights[id(pillar)] * pillar.probs
    evidence = _normalise_distribution(evidence)

    conflicts = _conflicts_for_supports(spatial, supports, policy)
    if spatial_anchor:
        final, protected = _protect_anchor_distribution(
            evidence,
            spatial.top_index,
            policy.protected_anchor_margin,
        )
        label = spatial.top_label
        decision_label = label
        status = "anchored_with_conflict" if conflicts or protected else "anchored"
        if protected:
            notes.append("A contradictory supporting vote was prevented from overriding the clear spatial anchor.")
        if conflicts:
            notes.append("One or more supporting pillars strongly disagree with the spatial anchor.")
        contributions[spatial.name] = _contribution(
            spatial,
            effective_weight=spatial_weight,
            state="spatial_anchor",
            used_for_valence=True,
            reason="Primary semantic signal; clear visual evidence anchors the final label.",
        )
        for pillar in supports:
            relation = "supporting" if pillar.top_index == spatial.top_index else "opposing"
            contributions[pillar.name] = _contribution(
                pillar,
                effective_weight=support_weights[id(pillar)],
                state=relation,
                used_for_valence=True,
                reason=(
                    "Bounded supporting evidence for the spatial anchor."
                    if relation == "supporting"
                    else "Bounded contradictory evidence; it is visible as a conflict and cannot replace a clear spatial label."
                ),
            )
    else:
        # We still expose a class-shaped aggregate so the user can inspect the
        # evidence, but make the decision explicitly provisional.  This avoids
        # calling a 34/33/33 spatial maximum a confident visual conclusion.
        final = evidence.copy()
        label = CLASS_NAMES[int(np.argmax(final))]
        top_margin = float(np.sort(final)[-1] - np.sort(final)[-2])
        status = "provisional" if top_margin >= policy.spatial_min_margin else "uncertain"
        decision_label = f"Provisional: {label}" if status == "provisional" else "Uncertain"
        notes.append("Spatial evidence is near-tied or below its declared anchor-quality threshold.")
        contributions[spatial.name] = _contribution(
            spatial,
            effective_weight=spatial_weight,
            state="spatial_provisional",
            used_for_valence=True,
            reason="Spatial score is visible, but not strong enough to anchor a final label.",
        )
        for pillar in supports:
            contributions[pillar.name] = _contribution(
                pillar,
                effective_weight=support_weights[id(pillar)],
                state="provisional_support",
                used_for_valence=True,
                reason="Bounded evidence used only while the visual decision remains provisional.",
            )

    return FusionResult(
        probabilities=final,
        evidence_probabilities=evidence,
        label=label,
        decision_label=decision_label,
        status=status,
        spatial_label=spatial.top_label,
        spatial_is_anchor=spatial_anchor,
        spatial_margin=spatial.top_two_margin,
        anchor_protection_applied=spatial_anchor and not np.allclose(final, evidence, rtol=0.0, atol=1e-12),
        conflicts=conflicts,
        contributions=contributions,
        structural_evidence=structural,
        notes=tuple(notes),
    )


# A short alias makes staged integration into the current app natural while
# retaining the explicit name in documentation and tests.
fuse_pillars = fuse_six_pillars


__all__ = [
    "CLASS_NAMES",
    "DEFAULT_BASE_WEIGHTS",
    "UNIFORM",
    "ConflictFlag",
    "FusionConfig",
    "FusionResult",
    "PillarContribution",
    "PillarResult",
    "coerce_pillar_result",
    "fuse_pillars",
    "fuse_six_pillars",
]
