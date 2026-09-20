"""Inference entry points.

The legacy frame/Whisper path has optional heavyweight dependencies.  Do not
import it eagerly: the canonical semantic five-pillar path deliberately works
without Whisper and should remain usable in a minimal inference environment.
"""

from .semantic_five_pillar_inference import (
    CheckpointContractError,
    SemanticFivePillarInferencer,
)

__all__ = [
    "CheckpointContractError",
    "SceneMotionInferencer",
    "SemanticFivePillarInferencer",
]


def __getattr__(name: str):
    """Load the optional legacy route only when a caller explicitly asks for it."""

    if name == "SceneMotionInferencer":
        from .video_inference import SceneMotionInferencer

        return SceneMotionInferencer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
