"""Adapter exposing the trained CLIP spatial head as pillar one.

The long-running LIRIS experiment saved a semantic five-pillar checkpoint
whose spatial branch consumes 512-D CLIP features and has its own auxiliary
classification head.  This adapter deliberately uses that *spatial head*, not
the checkpoint's full five-pillar prediction, so it can be the visual anchor
of the root six-pillar application without double-counting its audio, colour,
motion, or temporal branches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from scene_motion_llm.inference.semantic_five_pillar_inference import SemanticFivePillarInferencer


ROOT_CLASS_NAMES = ("Negative", "Neutral", "Positive")
SEMANTIC_CLASS_NAMES = ("Positive", "Neutral", "Negative")


def default_checkpoint_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "scene_motion_llm"
        / "checkpoints"
        / "semantic_five_pillar_v3_clip"
        / "best_semantic_five_pillar.pt"
    )


def _root_order(semantic_probs: np.ndarray) -> np.ndarray:
    values = np.asarray(semantic_probs, dtype=np.float32)
    if values.shape != (3,):
        raise ValueError("The trained spatial head must return three class probabilities.")
    by_name = {name: float(values[index]) for index, name in enumerate(SEMANTIC_CLASS_NAMES)}
    root_values = np.asarray([by_name[name] for name in ROOT_CLASS_NAMES], dtype=np.float32)
    return root_values / root_values.sum()


class TrainedSpatialPillar:
    """Run the trained v3 checkpoint's spatial auxiliary head for one video."""

    def __init__(self, checkpoint_path: str | Path | None = None, *, device: str | None = None) -> None:
        path = Path(checkpoint_path) if checkpoint_path is not None else default_checkpoint_path()
        self.inferencer = SemanticFivePillarInferencer(path, device=device)

    @torch.inference_mode()
    def analyze(self, video_path: str | Path) -> dict[str, Any]:
        """Return only trained spatial evidence in root-app class order.

        ``_prepare_model_inputs`` is used here to guarantee exactly the saved
        extractor and normalizer contract before the model's independent
        spatial head is evaluated.  The result excludes the checkpoint's final
        fusion prediction by design.
        """

        extraction = self.inferencer.extractor.extract(video_path)
        sequences, availability, reliability, sequence_length = self.inferencer._prepare_model_inputs(extraction)
        output = self.inferencer.model(sequences, availability, reliability)
        logits = output["pillar_logits"]["spatial"][0]
        semantic_probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        probs = _root_order(semantic_probs)
        reliability_value = float(output["pillar_reliability"][0, 2].detach().cpu().item())
        sorted_probs = np.sort(probs)
        margin = float(sorted_probs[-1] - sorted_probs[-2])
        top_index = int(np.argmax(probs))
        raw_spatial_reliability = np.asarray(extraction["reliability"]["spatial"], dtype=np.float32)
        return {
            "name": "Spatial",
            "probs": probs,
            "reliability": float(np.clip(reliability_value, 0.0, 1.0)),
            "available": bool(reliability_value > 0.0),
            "contributes_to_valence": True,
            # The trained head, rather than raw prompt margin, owns the visual
            # ranking.  A small margin is still exposed for the fusion layer to
            # decide whether to display an uncertainty warning.
            "status": "trained_spatial_anchor" if margin >= 0.05 else "trained_spatial_uncertain",
            "evidence": {
                "checkpoint": str(self.inferencer.checkpoint_path),
                "sequence_length": sequence_length,
                "top_label": ROOT_CLASS_NAMES[top_index],
                "top_two_margin": margin,
                "trained_spatial_reliability": reliability_value,
                "mean_extractor_reliability": float(raw_spatial_reliability.mean()) if raw_spatial_reliability.size else 0.0,
                "class_order_source": list(SEMANTIC_CLASS_NAMES),
                "note": "This is the trained checkpoint's spatial auxiliary head, not its full multimodal fusion output.",
            },
        }


__all__ = ["ROOT_CLASS_NAMES", "TrainedSpatialPillar", "default_checkpoint_path"]
