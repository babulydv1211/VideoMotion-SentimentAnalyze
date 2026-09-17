
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import torch

# Patch torch.classes.__path__ to suppress Streamlit module inspection warning
if hasattr(torch, "classes"):
    try:
        torch.classes.__path__ = []
    except Exception:
        pass
import tempfile
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Do not import the legacy model here.  It imports optional audio/text
# dependencies at module-load time, while the semantic five-pillar path does
# not need them until it actually analyses a soundtrack.  Keeping both imports
# lazy lets a trained semantic checkpoint open even in a lean inference env.

PROJECT_ROOT = Path(__file__).resolve().parent
# ``streamlit run scene_motion_llm/app.py`` already has this parent on
# sys.path.  Add it only when the script is launched directly from inside the
# package directory, so the lazy ``scene_motion_llm.*`` inference imports work
# in both supported launch forms.
PROJECT_PARENT = str(PROJECT_ROOT.parent)
if PROJECT_PARENT not in sys.path:
    sys.path.insert(0, PROJECT_PARENT)
SENTIMENT_LABELS = ("Positive", "Neutral", "Negative")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SceneMotion AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# MODERN GLASS UI
# =========================================================
st.markdown("""
<style>

.stApp { background: #f7f8fc; }
.block-container { max-width: 1080px; padding-top: 3rem; padding-bottom: 3rem; }

/* Glass card */
.glass { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.4rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); }

/* Title */
.title { font-size: 2.5rem; font-weight: 700; color: #172554; }

/* Subtitle */
.subtitle { color: #64748b; margin-bottom: 1.5rem; }

/* Sentiment styles */
.pos { color: #15803d; font-size: 1.75rem; font-weight: 700; }
.neu { color: #a16207; font-size: 1.75rem; font-weight: 700; }
.neg { color: #b91c1c; font-size: 1.75rem; font-weight: 700; }
div.stButton > button { background: #1d4ed8; color: white; border: 0; border-radius: 9px; font-weight: 600; }
div.stButton > button:hover { background: #1e40af; color: white; }

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">🎬 SceneMotion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a video and get a concise scene-motion sentiment prediction.</div>', unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "model" not in st.session_state:
    st.session_state.model = None

if "inferencer" not in st.session_state:
    st.session_state.inferencer = None

if "device" not in st.session_state:
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"

if "backend" not in st.session_state:
    st.session_state.backend = None


# =========================================================
# CHECKPOINT DISCOVERY
# =========================================================
def _existing_path(value: str | os.PathLike[str] | None) -> Path | None:
    """Return an existing file path without raising for an optional setting."""
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_file() else None


def find_semantic_checkpoint() -> Path | None:
    """Prefer the canonical five-pillar checkpoint, including an explicit path."""
    candidates: list[Path] = []
    explicit = _existing_path(os.environ.get("SCENEMOTION_SEMANTIC_CHECKPOINT"))
    if explicit is not None:
        candidates.append(explicit)

    checkpoint_root = PROJECT_ROOT / "checkpoints"
    candidates.extend(
        [
            # The v3 compact experiment is trained from the cache-repaired,
            # movie-disjoint workflow and should supersede the legacy v2 run
            # once it exists.
            checkpoint_root / "semantic_five_pillar_v3_compact" / "best_semantic_five_pillar.pt",
            checkpoint_root / "semantic_five_pillar" / "best_semantic_five_pillar.pt",
            checkpoint_root / "semantic_five_pillar" / "best_model.pt",
            checkpoint_root / "best_semantic_five_pillar.pt",
            checkpoint_root / "semantic_five_pillar.pt",
        ]
    )
    if checkpoint_root.is_dir():
        # This supports a renamed experiment checkpoint, while avoiding the
        # unrelated legacy checkpoints in the same tree.
        candidates.extend(sorted(checkpoint_root.glob("**/*semantic*.pt")))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate not in seen and candidate.is_file():
            seen.add(candidate)
            return candidate
    return None


def find_legacy_checkpoint() -> Path | None:
    """Return only the prior raw-video LIRIS baseline checkpoint."""
    return _existing_path(
        os.environ.get("SCENEMOTION_LEGACY_CHECKPOINT")
        or PROJECT_ROOT / "checkpoints" / "stage2_accede" / "best_model.pt"
    )


def _legacy_checkpoint_problem(checkpoint_path: Path) -> str | None:
    """Reject obviously incompatible legacy checkpoints before model loading."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        return f"could not read checkpoint metadata ({error})"

    if not isinstance(checkpoint, dict):
        return "does not contain the expected checkpoint metadata"
    task = checkpoint.get("task")
    if task not in (None, "liris_accede_video_sentiment"):
        return f"belongs to task {task!r}, not the legacy raw-video LIRIS task"

    state_dict = checkpoint.get("model_state_dict")
    fusion_weight = state_dict.get("fusion_layer.0.weight") if isinstance(state_dict, dict) else None
    if fusion_weight is None:
        return "is missing the legacy fusion-layer weights"
    # The current legacy class has spatial (512), temporal (512), audio (128),
    # and text (256) inputs.  A different width would fail load_state_dict.
    if getattr(fusion_weight, "shape", (None, None))[1] != 1408:
        return "uses a different legacy architecture (fusion width is not 1408)"
    return None

# =========================================================
# LOAD MODEL
# =========================================================
def load_model(allow_legacy: bool = False) -> bool:
    """Load semantic inference when trained; expose legacy only by choice."""
    if st.session_state.inferencer is not None:
        return True

    semantic_checkpoint = find_semantic_checkpoint()
    if semantic_checkpoint is not None:
        try:
            with st.spinner("Loading the five-pillar affect model..."):
                try:
                    from scene_motion_llm.inference.semantic_five_pillar_inference import (
                        SemanticFivePillarInferencer,
                    )
                except ModuleNotFoundError as error:
                    # Support the common ``streamlit run app.py`` launch from
                    # inside this project folder as well as package launches.
                    if error.name != "scene_motion_llm":
                        raise
                    from inference.semantic_five_pillar_inference import (
                        SemanticFivePillarInferencer,
                    )

                inferencer = SemanticFivePillarInferencer(
                    checkpoint_path=str(semantic_checkpoint),
                    device=st.session_state.device,
                )
            st.session_state.model = getattr(inferencer, "model", None)
            st.session_state.inferencer = inferencer
            st.session_state.backend = "semantic_five_pillar"
            st.success("✅ Five-pillar LIRIS affect model loaded")
            return True
        except Exception as error:
            st.error(
                "The semantic checkpoint was found but could not be loaded. "
                f"It was left untouched: {error}"
            )
            return False

    if not allow_legacy:
        legacy_checkpoint = find_legacy_checkpoint()
        if legacy_checkpoint is not None:
            st.warning(
                "The five-pillar checkpoint has not been trained yet. A legacy "
                "visual/motion baseline is available below, but it is not the "
                "new audio–color–spatial–motion–temporal model."
            )
        else:
            st.warning(
                "No trained five-pillar checkpoint was found. Train the semantic "
                "LIRIS model first, then restart this app."
            )
        return False

    legacy_checkpoint = find_legacy_checkpoint()
    if legacy_checkpoint is None:
        st.error("No legacy LIRIS checkpoint was found either.")
        return False

    problem = _legacy_checkpoint_problem(legacy_checkpoint)
    if problem is not None:
        st.error(f"The legacy checkpoint will not be loaded because it {problem}.")
        return False

    try:
        with st.spinner("Loading legacy visual/motion baseline..."):
            # These imports are deliberately inside the legacy-only branch.
            from scene_motion_llm.models.sentiment_classifier import SceneMotionLLMModel
            from scene_motion_llm.inference import SceneMotionInferencer

            model = SceneMotionLLMModel(
                spatial_feature_dim=512,
                motion_feature_dim=2,
                temporal_hidden_dim=256,
                attention_dim=128,
                fusion_dim=512,
                num_classes=3,
                max_frames=30,
                dropout=0.3,
                pretrained=False,
            )
            inferencer = SceneMotionInferencer(
                model=model,
                checkpoint_path=str(legacy_checkpoint),
                device=st.session_state.device,
                max_frames=30,
            )
        st.session_state.model = inferencer.model
        st.session_state.inferencer = inferencer
        st.session_state.backend = "legacy_visual_motion_baseline"
        st.warning("Using the legacy visual/motion baseline, not the five-pillar model.")
        return True
    except Exception as error:
        st.error(
            "The legacy baseline could not be started. It may require the optional "
            f"audio/text dependencies or cached model weights: {error}"
        )
        return False

# =========================================================
# SAFE RESULT HANDLER
# =========================================================
def _as_jsonable(value: Any) -> Any:
    """Convert tensor/NumPy-style inference values for the downloadable JSON."""
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return _as_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _probability_items(result: dict[str, Any]) -> list[tuple[str, float]]:
    """Accept either an ordered vector or a label-to-probability mapping."""
    raw_probabilities = result.get("probabilities", [])
    if isinstance(raw_probabilities, dict):
        return [
            (label, float(raw_probabilities.get(label, raw_probabilities.get(label.lower(), 0.0))))
            for label in SENTIMENT_LABELS
        ]

    values = _as_jsonable(raw_probabilities)
    if not isinstance(values, list):
        values = []
    return [
        (label, float(values[index]) if index < len(values) else 0.0)
        for index, label in enumerate(SENTIMENT_LABELS)
    ]


def _pillar_items(result: dict[str, Any]) -> list[tuple[str, float]]:
    """Read optional semantic fusion weights without assuming one report schema."""
    raw_weights = result.get("pillar_weights", result.get("fusion_weights", {}))
    if isinstance(raw_weights, dict):
        return [
            (name, float(raw_weights.get(name, raw_weights.get(name.lower(), 0.0))))
            for name in ("audio", "color", "spatial", "motion", "temporal")
            if name in raw_weights or name.lower() in raw_weights
        ]
    values = _as_jsonable(raw_weights)
    if isinstance(values, list) and len(values) == 5:
        return list(zip(("audio", "color", "spatial", "motion", "temporal"), map(float, values)))
    return []

# =========================================================
# DISPLAY RESULT
# =========================================================
def show_result(result):

    sentiment = result.get("sentiment", "Unknown")
    confidence = float(result.get("confidence", 0.0))
    probability_items = _probability_items(result)

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🎯 Prediction Result")

    if sentiment == "Positive":
        st.markdown(f'<div class="pos">{sentiment}</div>', unsafe_allow_html=True)
    elif sentiment == "Neutral":
        st.markdown(f'<div class="neu">{sentiment}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="neg">{sentiment}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("Confidence", f"{confidence:.2%}")
    col2.metric("Device", st.session_state.device.upper())

    st.subheader("Class probabilities")
    for label, probability in probability_items:
        clipped_probability = min(1.0, max(0.0, probability))
        st.write(label)
        st.progress(clipped_probability)
        st.write(f"{clipped_probability:.2%}")

    pillar_items = _pillar_items(result)
    if pillar_items:
        st.divider()
        st.subheader("Five-pillar contribution")
        audio_status = result.get("audio_status")
        if audio_status:
            st.caption(f"Audio status: {audio_status}")
        pillar_columns = st.columns(len(pillar_items))
        for column, (pillar, weight) in zip(pillar_columns, pillar_items):
            column.metric(pillar.title(), f"{weight:.1%}")

    model_evidence = result.get("model_evidence", {}) or {}
    evidence = result.get("evidence", {}) or {}
    if model_evidence or evidence:
        st.divider()
        st.subheader("Research evidence")
        evidence_columns = st.columns(4)
        if st.session_state.backend == "semantic_five_pillar":
            pillar_summary = result.get("pillar_summary", {})
            if not isinstance(pillar_summary, dict):
                pillar_summary = {}
            available_pillars = sum(
                bool(entry.get("available"))
                for entry in pillar_summary.values()
                if isinstance(entry, dict)
            )
            top_pillar = max(pillar_items, key=lambda item: item[1], default=("NIL", 0.0))
            evidence_columns[0].metric("Windows analyzed", result.get("sequence_length", 0))
            evidence_columns[1].metric("Available pillars", f"{available_pillars}/5")
            evidence_columns[2].metric("Audio", result.get("audio_status", "unavailable"))
            evidence_columns[3].metric("Top fusion", f"{top_pillar[0].title()} {top_pillar[1]:.1%}")
            st.caption(
                "Pillar measurements are descriptive evidence. The learned fusion "
                "relates them to LIRIS valence labels; a dark image or fast movement "
                "is not itself a negative or harmful judgment."
            )
        else:
            evidence_columns[0].metric("Frames analyzed", result.get("num_frames", 0))
            evidence_columns[1].metric(
                "Probability margin",
                f"{float(model_evidence.get('probability_margin', 0.0)):.2%}",
            )
            evidence_columns[2].metric("Mean motion", f"{float(evidence.get('mean_motion', 0.0)):.3f}")
            evidence_columns[3].metric("Frame change", f"{float(evidence.get('frame_change', 0.0)):.3f}")
            st.caption(
                "Top-attended frames are inspection evidence, not a causal explanation."
            )
        top_frames = model_evidence.get("top_attended_frames", [])
        if top_frames:
            st.dataframe(top_frames, hide_index=True, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# VIDEO ANALYSIS
# =========================================================
def analyze_video(video_file, allow_legacy: bool = False):

    suffix = Path(getattr(video_file, "name", "upload.mp4")).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        suffix = ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_file.read())
        path = tmp.name

    try:
        if not load_model(allow_legacy=allow_legacy):
            return

        with st.spinner("AI is analyzing video..."):

            result = st.session_state.inferencer.analyze_video(path)

        st.success("Analysis Complete")

        show_result(result)

        # =========================
        # SAFE OPTIONAL SECTIONS
        # =========================

        with st.expander("Technical report", expanded=False):
            st.text(result.get("report", "No report generated."))

        # DOWNLOAD
        download_data = {
            "sentiment": result.get("sentiment"),
            "confidence": float(result.get("confidence", 0)),
            "probabilities": _as_jsonable(result.get("probabilities", [])),
            "evidence": _as_jsonable(result.get("evidence", {})),
            "model_evidence": _as_jsonable(result.get("model_evidence", {})),
            "pillar_weights": _as_jsonable(result.get("pillar_weights", result.get("fusion_weights", {}))),
            "audio_status": result.get("audio_status"),
            "backend": st.session_state.backend,
            "timestamp": datetime.now().isoformat()
        }

        st.download_button(
            "⬇ Download JSON",
            json.dumps(download_data, indent=2),
            "result.json",
            "application/json"
        )

    except Exception as error:
        st.error(f"Analysis could not complete: {error}")

    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

# =========================================================
# UI SECTION
# =========================================================
left, right = st.columns([1.4, 1], gap="large")

semantic_checkpoint = find_semantic_checkpoint()
legacy_checkpoint = find_legacy_checkpoint()
use_legacy_baseline = False

with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Upload a short video")
    if semantic_checkpoint is not None:
        st.success("Five-pillar checkpoint found. Audio, color, spatial, motion, and temporal cues will be fused.")
    elif legacy_checkpoint is not None:
        st.warning(
            "The semantic five-pillar checkpoint is not trained yet. The available "
            "legacy model is a visual/motion baseline."
        )
        use_legacy_baseline = st.checkbox(
            "Use the legacy visual/motion baseline for this session",
            value=False,
            help="This preserves the prior model path; it is not the new five-pillar model.",
        )
    else:
        st.info("Train a semantic five-pillar LIRIS checkpoint before analysing uploads.")
    file = st.file_uploader("Supported: MP4, AVI, MOV, MKV", type=["mp4", "avi", "mov", "mkv"])

    if file:
        st.caption(f"Ready to analyze: {file.name} · {file.size / 1024 / 1024:.1f} MB")
        st.video(file)
        if st.button("Analyze video", type="primary", use_container_width=True):
            analyze_video(file, allow_legacy=use_legacy_baseline)
    else:
        st.info("Choose a video to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("How it works")
    if semantic_checkpoint is not None:
        st.markdown("""
        1. Upload a clip of up to 15 seconds.
        2. The model measures audio, color/lighting, spatial context, motion, and temporal change.
        3. Reliability-aware fusion predicts Positive, Neutral, or Negative with confidence.
        """)
    else:
        st.markdown("""
        1. Train the five-pillar LIRIS model to create its checkpoint.
        2. Restart this app so it can discover the checkpoint.
        3. The optional legacy baseline remains available for comparison.
        """)
    st.divider()
    st.caption(f"Running on {st.session_state.device.upper()} · Results stay on this device")
    st.markdown("</div>", unsafe_allow_html=True)
