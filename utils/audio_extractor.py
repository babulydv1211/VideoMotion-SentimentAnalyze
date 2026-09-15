import os
import torch
import librosa
import numpy as np
import logging
import whisper

logger = logging.getLogger(__name__)

class AudioExtractor:
    def __init__(self, sample_rate=16000, n_mfcc=13, max_audio_frames=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_audio_frames = max_audio_frames
        self.device = device
        
        # Load whisper base model
        try:
            # We use 'base' to balance speed and accuracy
            self.whisper_model = whisper.load_model('base', device=self.device)
            logger.info("Loaded Whisper 'base' model for audio transcription.")
        except Exception as e:
            logger.warning(f"Could not load Whisper model: {e}")
            self.whisper_model = None

    def extract_audio_features(self, video_path):
        """
        Extracts MFCCs and Text Transcript from a video.
        Returns:
            dict: {
                'mfcc': np.array of shape (n_mfcc, max_audio_frames),
                'text': str (transcript)
            }
            or None if no audio track exists.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        try:
            # Load audio using librosa directly from video file
            audio_array, _ = librosa.load(video_path, sr=self.sample_rate, mono=True)
            
            if audio_array is None or len(audio_array) == 0:
                return None
                
            # Ensure float32 for librosa and whisper
            audio_array = audio_array.astype(np.float32)
            
            # --- 1. Compute MFCCs ---
            mfccs = librosa.feature.mfcc(y=audio_array, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            
            # Pad or truncate MFCCs to fixed temporal size (max_audio_frames)
            if mfccs.shape[1] < self.max_audio_frames:
                pad_len = self.max_audio_frames - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_len)), mode='constant')
            else:
                mfccs = mfccs[:, :self.max_audio_frames]
            
            # --- 2. Transcribe Text ---
            text = ""
            if self.whisper_model is not None:
                audio_tensor = torch.from_numpy(audio_array).float()
                result = self.whisper_model.transcribe(audio_tensor, fp16=(torch.cuda.is_available() and self.device != 'cpu'))
                text = result.get('text', '').strip()
                
            return {
                'mfcc': mfccs,
                'text': text
            }
            
        except Exception as e:
            logger.debug(f"No audio track or audio extraction skipped for {video_path}: {e}")
            return None

