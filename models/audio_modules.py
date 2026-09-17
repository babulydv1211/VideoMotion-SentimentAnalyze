import logging

import torch
import torch.nn as nn

# transformers is optional — loaded lazily inside TextRoBERTa so that this
# module can be imported even when the package is not installed.

logger = logging.getLogger(__name__)

class AudioLSTM(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x is (Batch, MFCC_dim, Time) -> (B, 13, 100)
        # LSTM expects (Batch, Time, Feature)
        x = x.transpose(1, 2)
        out, (hn, cn) = self.lstm(x)
        # Use the last hidden state from both directions
        hn = torch.cat([hn[-2], hn[-1]], dim=1) 
        features = self.projection(hn)
        return features

class TextRoBERTa(nn.Module):
    def __init__(self, model_name='roberta-base', hidden_dim=256, dropout=0.3, freeze=True):
        super().__init__()
        try:
            from transformers import RobertaModel, RobertaTokenizer  # lazy — optional dep
            self.roberta = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded {model_name} for text sentiment.")

            if freeze:
                for param in self.roberta.parameters():
                    param.requires_grad = False

            self.projection = nn.Sequential(
                nn.Linear(self.roberta.config.hidden_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        except Exception as e:
            logger.warning(f"Could not load RoBERTa: {e}")
            self.roberta = None
            
    def forward(self, texts, device):
        if self.roberta is None or not texts:
            # Fallback tensor if RoBERTa failed
            return torch.zeros((len(texts) if texts else 1, 256), device=device)
            
        # Handle empty strings
        safe_texts = [t if t and len(t.strip()) > 0 else "empty" for t in texts]
        
        inputs = self.tokenizer(safe_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.roberta(**inputs)
        # Pooler output is the embedding for [CLS] token
        features = self.projection(outputs.pooler_output)
        return features
