"""Sequence-aware, reliability-masked five-pillar affect fusion.

This module is deliberately independent of feature extraction.  It consumes
already aligned PyTorch sequences from the five semantic pillars:

``audio``, ``color``, ``spatial``, ``motion``, and ``temporal``.

The expected input contract is::

    sequences[name]     # float tensor shaped (batch, steps, input_dims[name])
    availability[name]  # tensor shaped (batch, steps), 0 means unavailable
    reliability[name]   # optional tensor shaped (batch, steps), in [0, 1]

``availability`` may itself be continuous when a separate ``reliability``
mapping is not available.  When both are supplied, their product is the
effective reliability used by attention and fusion.  A mute video should pass
an all-zero audio availability/reliability mask; its audio fusion weight is
then exactly zero, rather than producing a NaN or a fabricated audio result.

The model predicts classes in ``(Positive, Neutral, Negative)`` order.  It
does *not* hard-code semantic rules such as "dark equals negative".  Instead,
the gates learn how much to trust each available pillar from training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# This is also the order used in every stacked output tensor, including
# ``fusion_weights`` and ``pillar_reliability``.
PILLAR_NAMES: Final[tuple[str, ...]] = (
    "audio",
    "color",
    "spatial",
    "motion",
    "temporal",
)
SOURCE_PILLAR_NAMES: Final[tuple[str, ...]] = PILLAR_NAMES[:-1]
CLASS_NAMES: Final[tuple[str, str, str]] = ("Positive", "Neutral", "Negative")


# Defaults match ``utils.semantic_pillars``.  Callers can supply a different
# mapping in SemanticFivePillarConfig when using another extractor.
DEFAULT_INPUT_DIMS: Final[dict[str, int]] = {
    "audio": 20,
    "color": 16,
    "spatial": 24,
    "motion": 14,
    "temporal": 12,
}


@dataclass(frozen=True)
class SemanticFivePillarConfig:
    """Architecture and input-dimension contract for the fusion model.

    ``input_dims`` must contain exactly the five keys in :data:`PILLAR_NAMES`.
    Every sequence passed to :class:`SemanticFivePillarModel` has the matching
    final dimension.  The default dimensions match the companion semantic
    feature extractor, but they are not coupled by an import so this model has
    no OpenCV, audio, or transformer dependency.

    ``modality_dropout`` is a training-time robustness hook.  It randomly
    removes each source pillar (audio/color/spatial/motion) for a sample while
    leaving the derived temporal branch as an anchor.  Set it to zero for
    deterministic training and inference; a modest value such as 0.10--0.20
    helps train a model that remains usable when an uploaded video is mute.
    """

    input_dims: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_INPUT_DIMS))
    embedding_dim: int = 128
    attention_dim: int = 64
    temporal_hidden_dim: int = 128
    fusion_hidden_dim: int = 256
    dropout: float = 0.20
    modality_dropout: float = 0.0
    num_classes: int = 3
    class_names: tuple[str, str, str] = CLASS_NAMES

    def __post_init__(self) -> None:
        dimensions = dict(self.input_dims)
        missing = [name for name in PILLAR_NAMES if name not in dimensions]
        extras = [name for name in dimensions if name not in PILLAR_NAMES]
        if missing or extras:
            raise ValueError(
                "input_dims must contain exactly "
                f"{PILLAR_NAMES}; missing={missing}, extra={extras}."
            )
        dimensions = {name: int(dimensions[name]) for name in PILLAR_NAMES}
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError(f"All input dimensions must be positive: {dimensions}.")
        if self.embedding_dim <= 0 or self.attention_dim <= 0:
            raise ValueError("embedding_dim and attention_dim must be positive.")
        if self.temporal_hidden_dim <= 0 or self.fusion_hidden_dim <= 0:
            raise ValueError("temporal_hidden_dim and fusion_hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1).")
        if self.num_classes != 3 or len(self.class_names) != 3:
            raise ValueError(
                "This P/N/N model has exactly three outputs in Positive, Neutral, "
                "Negative order."
            )
        # Store an ordered ordinary dict so serialised configs and model inputs
        # have a stable, human-readable contract.
        object.__setattr__(self, "input_dims", dimensions)


def _finite_reliability(mask: Tensor, *, name: str) -> Tensor:
    """Return a finite [0, 1] reliability tensor without mutating the input.

    Non-finite values are treated as unavailable.  Clamping makes the forward
    pass robust to imperfect decoder metadata while keeping the public contract
    simple: callers should normally provide values in [0, 1].
    """

    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, received {type(mask)!r}.")
    if mask.dtype == torch.bool:
        return mask.to(dtype=torch.float32)
    if not (mask.is_floating_point() or mask.is_complex()):
        return mask.to(dtype=torch.float32).clamp_(0.0, 1.0)
    if mask.is_complex():
        raise TypeError(f"{name} must be real-valued, not complex.")
    return torch.nan_to_num(mask.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _reliability_softmax(logits: Tensor, reliability: Tensor, dim: int) -> Tensor:
    """Softmax that assigns zero mass to zero-reliability entries.

    Reliability is added in log-space, so a 0.2-quality pillar is naturally
    less competitive than an equally scored 1.0-quality pillar.  An all-zero
    row returns all zeros instead of the NaNs produced by a normal masked
    softmax over only ``-inf`` values.
    """

    reliability = reliability.to(dtype=logits.dtype).clamp(0.0, 1.0)
    valid = reliability > 0
    safe_logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    safe_logits = safe_logits + torch.log(reliability.clamp_min(torch.finfo(logits.dtype).eps))
    safe_logits = safe_logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(safe_logits, dim=dim) * valid.to(dtype=logits.dtype)
    denominator = weights.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0,
        weights / denominator.clamp_min(torch.finfo(logits.dtype).eps),
        torch.zeros_like(weights),
    )


class _MaskedAttentionPool(nn.Module):
    """Reliability-aware attention pooling for one ordered sequence."""

    def __init__(self, feature_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, values: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scores = self.scorer(values).squeeze(-1)
        weights = _reliability_softmax(scores, reliability, dim=1)
        pooled = torch.sum(values * weights.unsqueeze(-1), dim=1)
        # Returning a finite, masked score map keeps report rendering simple.
        masked_scores = torch.where(reliability > 0, scores, torch.zeros_like(scores))
        return pooled, weights, masked_scores


class _PillarSequenceEncoder(nn.Module):
    """Project one B,T,D semantic sequence and pool it over valid windows."""

    def __init__(self, input_dim: int, embedding_dim: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )
        self.pool = _MaskedAttentionPool(embedding_dim, attention_dim)

    def forward(self, values: Tensor, reliability: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = torch.nan_to_num(values, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        encoded = self.project(values)
        # Suppress projection biases at missing windows before attention and
        # before the ordered temporal branch sees the sequence.
        encoded = encoded * (reliability > 0).to(dtype=encoded.dtype).unsqueeze(-1)
        pooled, weights, scores = self.pool(encoded, reliability)
        return encoded, pooled, weights, scores


class SemanticFivePillarModel(nn.Module):
    """Fuse five semantic video pillars into Positive/Neutral/Negative logits.

    Args:
        config: A :class:`SemanticFivePillarConfig` defining all sequence
            dimensions.  Construct the same config for training and inference
            and save it alongside ``state_dict``.

    Forward arguments:
        sequences: Mapping with all five names in :data:`PILLAR_NAMES`, each a
            ``(B, T, D_name)`` tensor.  ``audio`` may instead be ``None``;
            this is treated as a zero audio sequence if another pillar defines
            B and T.
        availability: Mapping with ``(B, T)`` hard masks.  Zero means there is
            no observation (for example, mute audio or padded windows).
        reliability: Optional mapping with ``(B, T)`` quality weights in
            ``[0, 1]``.  It is multiplied by availability.  This lets a caller
            mark low-detail/black visual windows as less reliable without
            inventing a sentiment label.

    Returns:
        A dictionary with final ``logits``, ``probabilities`` and
        ``predicted_class``; ``pillar_logits`` for all five auxiliary heads;
        global ``fusion_weights``/``fusion_mask``; and both per-pillar and
        ordered-fusion temporal attention weights.  The class order is
        :data:`CLASS_NAMES`.
    """

    def __init__(self, config: SemanticFivePillarConfig | None = None) -> None:
        super().__init__()
        self.config = config or SemanticFivePillarConfig()
        self.input_dims = self.config.input_dims
        self.embedding_dim = self.config.embedding_dim
        self.num_classes = self.config.num_classes
        self.class_names = self.config.class_names
        self.pillar_names = PILLAR_NAMES

        self.pillar_encoders = nn.ModuleDict(
            {
                name: _PillarSequenceEncoder(
                    self.input_dims[name],
                    self.embedding_dim,
                    self.config.attention_dim,
                    self.config.dropout,
                )
                for name in PILLAR_NAMES
            }
        )

        # First fuse semantic pillars at each ordered time window, then run a
        # sequence encoder.  This is the fifth/temporal branch's learned view
        # of how the evidence evolves over the entire clip.
        self.frame_gate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.config.attention_dim),
            nn.GELU(),
            nn.Linear(self.config.attention_dim, 1),
        )
        self.temporal_input = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
        )
        self.temporal_gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.config.temporal_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_output = nn.Sequential(
            nn.Linear(self.config.temporal_hidden_dim * 2, self.embedding_dim),
            nn.GELU(),
        )
        self.temporal_pool = _MaskedAttentionPool(self.embedding_dim, self.config.attention_dim)
        self.temporal_merge = nn.Sequential(
            nn.LayerNorm(self.embedding_dim * 2),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
        )

        self.pillar_heads = nn.ModuleDict(
            {name: nn.Linear(self.embedding_dim, self.num_classes) for name in PILLAR_NAMES}
        )
        gate_input_dim = self.embedding_dim * len(PILLAR_NAMES) + len(PILLAR_NAMES)
        self.fusion_gate = nn.Sequential(
            nn.Linear(gate_input_dim, self.config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.fusion_hidden_dim, len(PILLAR_NAMES)),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.config.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.fusion_hidden_dim, self.num_classes),
        )

    @property
    def modality_dropout(self) -> float:
        """Configured source-pillar dropout probability (active only in train mode)."""

        return self.config.modality_dropout

    def _model_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _prepare_inputs(
        self,
        sequences: Mapping[str, Tensor | None],
        availability: Mapping[str, Tensor],
        reliability: Mapping[str, Tensor] | None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        """Validate the public tensor contract and create effective masks."""

        if not isinstance(sequences, Mapping):
            raise TypeError("sequences must be a mapping keyed by the five pillar names.")
        if not isinstance(availability, Mapping):
            raise TypeError("availability must be a mapping keyed by the five pillar names.")
        if reliability is not None and not isinstance(reliability, Mapping):
            raise TypeError("reliability must be a mapping keyed by the five pillar names.")

        missing_sequences = [name for name in PILLAR_NAMES if name not in sequences]
        # Audio is intentionally allowed to be omitted/None for mute input.
        if missing_sequences and missing_sequences != ["audio"]:
            raise KeyError(f"sequences is missing required pillar(s): {missing_sequences}.")

        reference: Tensor | None = None
        for name in PILLAR_NAMES:
            value = sequences.get(name)
            if isinstance(value, Tensor):
                reference = value
                break
        if reference is None:
            raise ValueError("At least one pillar sequence tensor is required to define B and T.")
        if reference.ndim != 3:
            raise ValueError(
                f"Pillar sequences must have shape (B, T, D); got {tuple(reference.shape)}."
            )
        batch_size, steps = reference.shape[:2]
        dtype = self._model_dtype()

        prepared_sequences: dict[str, Tensor] = {}
        prepared_availability: dict[str, Tensor] = {}
        effective_reliability: dict[str, Tensor] = {}
        for name in PILLAR_NAMES:
            value = sequences.get(name)
            if value is None:
                if name != "audio":
                    raise KeyError(f"sequences['{name}'] may not be None.")
                value = torch.zeros(
                    batch_size,
                    steps,
                    self.input_dims[name],
                    device=reference.device,
                    dtype=dtype,
                )
            if not isinstance(value, Tensor):
                raise TypeError(f"sequences['{name}'] must be a torch.Tensor or None for audio.")
            expected_shape = (batch_size, steps, self.input_dims[name])
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"sequences['{name}'] must have shape {expected_shape}, got {tuple(value.shape)}."
                )
            value = value.to(dtype=dtype)
            prepared_sequences[name] = value

            raw_mask = availability.get(name)
            if raw_mask is None:
                if name == "audio" and sequences.get("audio") is None:
                    raw_mask = torch.zeros(batch_size, steps, device=value.device, dtype=torch.float32)
                else:
                    raise KeyError(f"availability is missing '{name}'.")
            if tuple(raw_mask.shape) != (batch_size, steps):
                raise ValueError(
                    f"availability['{name}'] must have shape {(batch_size, steps)}, "
                    f"got {tuple(raw_mask.shape)}."
                )
            available = _finite_reliability(raw_mask.to(device=value.device), name=f"availability['{name}']")
            prepared_availability[name] = available

            raw_reliability = reliability.get(name) if reliability is not None else None
            if raw_reliability is None:
                quality = torch.ones_like(available)
            else:
                if tuple(raw_reliability.shape) != (batch_size, steps):
                    raise ValueError(
                        f"reliability['{name}'] must have shape {(batch_size, steps)}, "
                        f"got {tuple(raw_reliability.shape)}."
                    )
                quality = _finite_reliability(
                    raw_reliability.to(device=value.device), name=f"reliability['{name}']"
                )
            effective_reliability[name] = (available * quality).to(dtype=dtype)

        return prepared_sequences, prepared_availability, effective_reliability

    def _apply_training_modality_dropout(
        self, effective_reliability: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Randomly hide source pillars during training, never at evaluation."""

        batch_size = next(iter(effective_reliability.values())).shape[0]
        device = next(iter(effective_reliability.values())).device
        keep = torch.ones(batch_size, len(PILLAR_NAMES), device=device)
        if self.training and self.modality_dropout > 0:
            source_keep = torch.rand(
                batch_size, len(SOURCE_PILLAR_NAMES), device=device
            ) >= self.modality_dropout
            keep[:, : len(SOURCE_PILLAR_NAMES)] = source_keep.to(dtype=keep.dtype)
        dropped = {
            name: effective_reliability[name] * keep[:, index].unsqueeze(-1)
            for index, name in enumerate(PILLAR_NAMES)
        }
        return dropped, keep

    def _encode_ordered_temporal_context(
        self, frame_embedding: Tensor, frame_reliability: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run a packed BiGRU and attention over the learned ordered fusion."""

        batch_size, steps, _ = frame_embedding.shape
        temporal_input = self.temporal_input(frame_embedding)
        # These masks are expected to describe valid prefixes followed by
        # padding.  This is the normal result of batched video-window padding.
        lengths = (frame_reliability > 0).sum(dim=1)
        encoded = temporal_input.new_zeros(
            batch_size, steps, self.config.temporal_hidden_dim * 2
        )
        usable_indices = torch.nonzero(lengths > 0, as_tuple=False).squeeze(-1)
        if usable_indices.numel() > 0:
            usable_input = temporal_input.index_select(0, usable_indices)
            usable_lengths = lengths.index_select(0, usable_indices).to(device="cpu", dtype=torch.long)
            packed = pack_padded_sequence(
                usable_input,
                usable_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.temporal_gru(packed)
            usable_output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=steps,
            )
            encoded = encoded.index_copy(0, usable_indices, usable_output)
        encoded = self.temporal_output(encoded)
        encoded = encoded * (frame_reliability > 0).to(dtype=encoded.dtype).unsqueeze(-1)
        pooled, weights, scores = self.temporal_pool(encoded, frame_reliability)
        return encoded, pooled, weights, scores

    def forward(
        self,
        sequences: Mapping[str, Tensor | None],
        availability: Mapping[str, Tensor],
        reliability: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Predict P/N/N logits from aligned five-pillar semantic sequences."""

        sequences, raw_availability, effective_reliability = self._prepare_inputs(
            sequences, availability, reliability
        )
        effective_reliability, dropout_keep = self._apply_training_modality_dropout(
            effective_reliability
        )

        sequence_embeddings: dict[str, Tensor] = {}
        pillar_embeddings: dict[str, Tensor] = {}
        pillar_attention_weights: dict[str, Tensor] = {}
        pillar_attention_scores: dict[str, Tensor] = {}
        for name in PILLAR_NAMES:
            encoded, pooled, weights, scores = self.pillar_encoders[name](
                sequences[name], effective_reliability[name]
            )
            sequence_embeddings[name] = encoded
            pillar_embeddings[name] = pooled
            pillar_attention_weights[name] = weights
            pillar_attention_scores[name] = scores

        stacked_sequence_embeddings = torch.stack(
            [sequence_embeddings[name] for name in PILLAR_NAMES], dim=2
        )  # (B, T, 5, E)
        stacked_reliability = torch.stack(
            [effective_reliability[name] for name in PILLAR_NAMES], dim=2
        )  # (B, T, 5)
        frame_gate_logits = self.frame_gate(stacked_sequence_embeddings).squeeze(-1)
        frame_pillar_weights = _reliability_softmax(
            frame_gate_logits, stacked_reliability, dim=2
        )
        frame_embedding = torch.sum(
            stacked_sequence_embeddings * frame_pillar_weights.unsqueeze(-1), dim=2
        )
        # A derived temporal sequence is meaningful wherever any pillar is
        # observed.  With normal input, the explicit temporal pillar ensures
        # this is true for every non-padding video window.
        frame_reliability = stacked_reliability.max(dim=2).values
        temporal_sequence, temporal_context, temporal_weights, temporal_scores = (
            self._encode_ordered_temporal_context(frame_embedding, frame_reliability)
        )

        # Blend the extractor's temporal descriptors with the learned ordered
        # cross-pillar context.  Thus temporal remains useful even if a caller
        # cannot provide a dedicated temporal descriptor sequence.
        pillar_embeddings["temporal"] = self.temporal_merge(
            torch.cat([pillar_embeddings["temporal"], temporal_context], dim=-1)
        )

        # Source-pillar quality is its mean effective window reliability.  The
        # temporal pillar uses the quality of the ordered evidence from which
        # it is derived, not merely its raw temporal descriptor availability.
        pillar_reliability = torch.stack(
            [
                effective_reliability[name].mean(dim=1)
                if name != "temporal"
                else frame_reliability.mean(dim=1)
                for name in PILLAR_NAMES
            ],
            dim=1,
        )
        pillar_available = pillar_reliability > 0
        stacked_pillar_embeddings = torch.stack(
            [pillar_embeddings[name] for name in PILLAR_NAMES], dim=1
        )
        gate_input = torch.cat(
            [stacked_pillar_embeddings.flatten(start_dim=1), pillar_reliability], dim=1
        )
        fusion_logits = self.fusion_gate(gate_input)
        fusion_weights = _reliability_softmax(fusion_logits, pillar_reliability, dim=1)
        fused_embedding = torch.sum(
            stacked_pillar_embeddings * fusion_weights.unsqueeze(-1), dim=1
        )
        logits = self.classifier(fused_embedding)
        probabilities = torch.softmax(logits, dim=1)

        pillar_logits: dict[str, Tensor] = {}
        for index, name in enumerate(PILLAR_NAMES):
            local_logits = self.pillar_heads[name](pillar_embeddings[name])
            # An unavailable head is deliberately zeroed so a UI cannot mistake
            # its classifier bias for evidence from a mute/absent modality.
            pillar_logits[name] = local_logits * pillar_available[:, index].to(
                dtype=local_logits.dtype
            ).unsqueeze(-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": probabilities.argmax(dim=1),
            "class_names": self.class_names,
            "pillar_logits": pillar_logits,
            # Alias makes the intent obvious to training/report code without
            # requiring a second computation.
            "per_pillar_logits": pillar_logits,
            "pillar_embeddings": pillar_embeddings,
            "fusion_weights": fusion_weights,
            "fusion_mask": pillar_available,
            "pillar_reliability": pillar_reliability,
            "frame_pillar_weights": frame_pillar_weights,
            "temporal_attention_weights": temporal_weights,
            "temporal_attention_scores": temporal_scores,
            "pillar_temporal_attention_weights": pillar_attention_weights,
            "pillar_temporal_attention_scores": pillar_attention_scores,
            "temporal_sequence": temporal_sequence,
            "temporal_features": temporal_context,
            "fused_features": fused_embedding,
            "availability": raw_availability,
            "effective_reliability": effective_reliability,
            "modality_dropout_keep": dropout_keep,
        }


# A descriptive alias for callers that prefer to name the fusion operation.
SemanticFivePillarFusionModel = SemanticFivePillarModel


__all__ = [
    "CLASS_NAMES",
    "DEFAULT_INPUT_DIMS",
    "PILLAR_NAMES",
    "SOURCE_PILLAR_NAMES",
    "SemanticFivePillarConfig",
    "SemanticFivePillarFusionModel",
    "SemanticFivePillarModel",
]
