"""Sentiment model for pre-extracted CMU-MOSEI sequence features."""

import torch
import torch.nn as nn

from .attention import TemporalAttention


class FeatureSentimentModel(nn.Module):
    """Classify sentiment from temporal feature sequences."""

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        attention_dim=128,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_norm = nn.LayerNorm(input_dim)

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        encoded_dim = hidden_dim * 2
        self.attention = TemporalAttention(
            feature_dim=encoded_dim,
            attention_dim=attention_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features, mask=None):
        features = self.input_norm(features.float())
        encoded, _ = self.encoder(features)
        pooled, attention_weights, attention_scores = self.attention(encoded, mask=mask)
        logits = self.classifier(pooled)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "attention_weights": attention_weights,
            "attention_scores": attention_scores,
            "features": pooled
        }
