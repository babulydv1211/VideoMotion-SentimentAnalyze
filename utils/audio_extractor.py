"""Legacy audio feature extractor using librosa (MFCCs) and optionally Whisper.

Whisper is fully optional — if it cannot be imported (e.g. blocked by a
system Application Control policy or simply not installed), the extractor
continues with MFCC features only and returns an empty transcript.
"""

import logging
import os

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Whisper — imported once at module load, failures are remembered
# ---------------------------------------------------------------------------
_WHISPER_MODULE = None
_WHISPER_CHECKED = False


def _try_import_whisper():
    """Return the whisper module or None if it cannot be loaded."""
    global _WHISPER_MODULE, _WHISPER_CHECKED
    if _WHISPER_CHECKED:
        return _WHISPER_MODULE
    _WHISPER_CHECKED = True
    try:
        import whisper  # noqa: PLC0415
        _WHISPER_MODULE = whisper
        logger.info("Whisper is available for transcript extraction.")
    except Exception as exc:
        logger.warning(
            "Whisper could not be imported (%s). "
            "Transcription will be skipped; MFCC features remain available.",
            exc,
        )
        _WHISPER_MODULE = None
    return _WHISPER_MODULE


class AudioExtractor:
    """Extract MFCCs and (optionally) a Whisper transcript from a video."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mfcc: int = 13,
        max_audio_frames: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_audio_frames = max_audio_frames
        self.device = device
        self.whisper_model = None

        whisper = _try_import_whisper()
        if whisper is not None:
            try:
                self.whisper_model = whisper.load_model("base", device=self.device)
                logger.info("Loaded Whisper 'base' model for audio transcription.")
            except Exception as exc:
                logger.warning("Whisper model could not load (%s). Continuing without transcription.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_audio_features(self, video_path: str):
        """Extract MFCCs and optional transcript from a video file.

        Returns a dict ``{'mfcc': np.ndarray, 'text': str}`` or ``None``
        when the file has no decodable audio track.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        try:
            audio_array, _ = librosa.load(video_path, sr=self.sample_rate, mono=True)
        except Exception as exc:
            logger.debug("No audio track or decode failure for %s: %s", video_path, exc)
            return None

        if audio_array is None or len(audio_array) == 0:
            return None

        audio_array = audio_array.astype(np.float32)

        # --- MFCCs (always available) ---
        mfccs = librosa.feature.mfcc(y=audio_array, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        if mfccs.shape[1] < self.max_audio_frames:
            pad_len = self.max_audio_frames - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_len)), mode="constant")
        else:
            mfccs = mfccs[:, : self.max_audio_frames]

        # --- Transcript (optional — skipped gracefully if Whisper unavailable) ---
        text = ""
        if self.whisper_model is not None:
            try:
                audio_tensor = torch.from_numpy(audio_array).float()
                fp16 = torch.cuda.is_available() and self.device != "cpu"
                result = self.whisper_model.transcribe(audio_tensor, fp16=fp16)
                text = result.get("text", "").strip()
            except Exception as exc:
                logger.debug("Whisper transcription skipped: %s", exc)

        return {"mfcc": mfccs, "text": text}
