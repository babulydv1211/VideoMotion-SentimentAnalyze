
import torch
import torch.nn as nn

from .spatial_extractor import SpatialFeatureExtractor
from .temporal_motion import TemporalMotionLSTM
from .attention import TemporalAttention, MultiHeadAttention
from .audio_modules import AudioLSTM, TextRoBERTa


# =========================================================
# SENTIMENT CLASSIFIER
# =========================================================

class SentimentClassifier(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_classes=3,
        dropout=0.3
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim // 2,
                num_classes
            )
        )

    def forward(self, x):

        return self.classifier(x)


# =========================================================
# MAIN MODEL
# =========================================================

class SceneMotionLLMModel(nn.Module):

    def __init__(
        self,
        spatial_feature_dim=512,
        motion_feature_dim=2,
        temporal_hidden_dim=256,
        attention_dim=128,
        fusion_dim=512,
        num_classes=3,
        max_frames=30,
        dropout=0.3,
        pretrained=True
    ):

        super().__init__()

        self.temporal_hidden_dim = temporal_hidden_dim
        self.max_frames = max_frames

        # =================================================
        # SPATIAL FEATURE EXTRACTOR
        # =================================================

        self.spatial_extractor = SpatialFeatureExtractor(
            pretrained=pretrained,
            feature_dim=spatial_feature_dim
        )

        # =================================================
        # MOTION FEATURE PROJECTION
        # =================================================

        self.motion_projection = nn.Sequential(

            nn.Linear(
                motion_feature_dim,
                temporal_hidden_dim
            ),

            nn.ReLU(inplace=True)
        )

        # =================================================
        # TEMPORAL LSTM
        # =================================================

        self.temporal_lstm = TemporalMotionLSTM(
            input_dim=temporal_hidden_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        # =================================================
        # TEMPORAL ATTENTION
        # =================================================

        self.temporal_attention = TemporalAttention(
            feature_dim=temporal_hidden_dim * 2,
            attention_dim=attention_dim
        )

        # =================================================
        # MULTI-HEAD SPATIAL ATTENTION
        # =================================================

        self.multi_head_attention = MultiHeadAttention(
            feature_dim=spatial_feature_dim,
            num_heads=4
        )

        # =================================================
        # AUDIO & TEXT FEATURES (Multimodal)
        # =================================================
        
        self.audio_hidden_dim = 128
        self.text_hidden_dim = 256
        
        self.audio_lstm = AudioLSTM(hidden_dim=self.audio_hidden_dim, dropout=dropout)
        self.text_roberta = TextRoBERTa(hidden_dim=self.text_hidden_dim, dropout=dropout)


        # =================================================
        # FEATURE FUSION
        # =================================================

        fusion_input_dim = (
            spatial_feature_dim +
            (temporal_hidden_dim * 2) +
            self.audio_hidden_dim +
            self.text_hidden_dim
        )

        self.fusion_layer = nn.Sequential(

            nn.Linear(
                fusion_input_dim,
                fusion_dim
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.LayerNorm(fusion_dim)
        )

        # =================================================
        # FINAL CLASSIFIER
        # =================================================

        self.sentiment_classifier = SentimentClassifier(
            input_dim=fusion_dim,
            hidden_dim=256,
            num_classes=num_classes,
            dropout=dropout
        )

    # =====================================================
    # FORWARD PASS
    # =====================================================

    def forward(
        self,
        frames,
        optical_flow=None,
        audio_features=None,
        mask=None
    ):

        """
        frames:
            (B, T, 3, H, W)

        optical_flow:
            (B, T, 2, H, W)
        """

        B, T, C, H, W = frames.shape

        # =================================================
        # SPATIAL FEATURES
        # =================================================

        spatial_features = self.spatial_extractor(
            frames
        )

        # Shape:
        # (B, T, 512)

        # =================================================
        # MOTION FEATURES
        # =================================================

        if optical_flow is not None:

            # Convert:
            # (B, T, 2, H, W)
            # ->
            # (B, T, 2)

            motion_feats = optical_flow.mean(
                dim=(3, 4)
            )

            # Ensure frame count matches

            if motion_feats.shape[1] < T:

                padding = torch.zeros(
                    B,
                    T - motion_feats.shape[1],
                    motion_feats.shape[2],
                    device=motion_feats.device
                )

                motion_feats = torch.cat(
                    [motion_feats, padding],
                    dim=1
                )

            motion_feats = self.motion_projection(
                motion_feats
            )

        else:

            motion_feats = torch.zeros(
                B,
                T,
                self.temporal_hidden_dim,
                device=frames.device
            )

        # =================================================
        # TEMPORAL LSTM
        # =================================================

        temporal_features = self.temporal_lstm(
            motion_feats
        )

        # Shape:
        # (B, T, hidden_dim * 2)

        # =================================================
        # TEMPORAL ATTENTION
        # =================================================

        attended_motion, attention_weights, attention_scores = \
            self.temporal_attention(
                temporal_features,
                mask=mask
            )

        # =================================================
        # SPATIAL ATTENTION
        # =================================================

        spatial_attended = self.multi_head_attention(
            values=spatial_features,
            keys=spatial_features,
            query=spatial_features,
            mask=None
        )

        # =================================================
        # GLOBAL SPATIAL POOLING
        # =================================================

        if mask is not None:

            masked_spatial = (
                spatial_attended *
                mask.unsqueeze(-1)
            )

            avg_spatial = masked_spatial.sum(dim=1) / (
                mask.sum(dim=1, keepdim=True) + 1e-8
            )

        else:

            avg_spatial = spatial_attended.mean(dim=1)

        # =================================================
        # AUDIO & TEXT PROCESSING
        # =================================================
        
        if audio_features is not None and 'mfcc' in audio_features:
            mfcc = audio_features['mfcc'].to(frames.device)
            texts = audio_features.get('text', [])
            
            # Forward Audio
            audio_feats = self.audio_lstm(mfcc)
            
            # Forward Text
            text_feats = self.text_roberta(texts, frames.device)
        else:
            # Fallback for silent video / visual-only mode
            audio_feats = torch.zeros((B, self.audio_hidden_dim), device=frames.device)
            text_feats = torch.zeros((B, self.text_hidden_dim), device=frames.device)

        # =================================================
        # FEATURE FUSION
        # =================================================

        fused_features = torch.cat(
            [
                avg_spatial,
                attended_motion,
                audio_feats,
                text_feats
            ],
            dim=1
        )

        fused_features = self.fusion_layer(
            fused_features
        )

        # =================================================
        # CLASSIFICATION
        # =================================================

        logits = self.sentiment_classifier(
            fused_features
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probs,
            dim=1
        )

        # =================================================
        # OUTPUT
        # =================================================

        return {

            "logits": logits,

            "probabilities": probs,

            "predicted_class": predictions,

            "attention_weights": attention_weights,

            "spatial_features": avg_spatial,

            "temporal_features": attended_motion,

            "fused_features": fused_features,
            "audio_features": audio_feats,
            "text_features": text_feats
        }

    # =====================================================
    # FREEZE / UNFREEZE
    # =====================================================

    def freeze_spatial_extractor(self):

        for param in self.spatial_extractor.parameters():

            param.requires_grad = False

    def unfreeze_spatial_extractor(self):

        for param in self.spatial_extractor.parameters():

            param.requires_grad = True