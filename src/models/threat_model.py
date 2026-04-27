"""
threat_model.py
Ensemble fusion of BiLSTM encoder + CNN classifier for multi-class
threat detection.

Research question addressed:
    Does fusing temporal (LSTM) and spatial (CNN) representations of
    network flow data yield higher F1 scores than either architecture
    alone on the CICIDS2017 benchmark?

    H0: Hybrid F1 ≤ max(LSTM F1, CNN F1)
    H1: Hybrid F1 > max(LSTM F1, CNN F1)  ← what we test in ablation
"""

import torch
import torch.nn as nn
from src.models.lstm_encoder import BiLSTMEncoder
from src.models.cnn_classifier import CNNClassifier


class ThreatDetectionModel(nn.Module):
    """
    BiLSTM + CNN ensemble with a fully-connected classification head.

    Inputs:
        x_seq  : (batch, window_size, input_dim)  — for LSTM
        x_feat : (batch, input_dim)               — for CNN (last frame of window)

    Output:
        logits : (batch, num_classes)
    """

    def __init__(self, input_dim: int = 78, hidden_dim: int = 256,
                 num_layers: int = 2, num_filters: int = 128,
                 num_classes: int = 15, dropout: float = 0.3):
        super().__init__()

        self.lstm_enc = BiLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.cnn_cls = CNNClassifier(
            input_dim=input_dim,
            num_filters=num_filters,
            dropout=dropout,
        )

        fused_dim = self.lstm_enc.output_dim + self.cnn_cls.output_dim  # 512 + 128 = 640

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_seq: torch.Tensor,
                x_feat: torch.Tensor = None) -> torch.Tensor:
        """
        If x_feat is None, use the last frame of x_seq for the CNN branch.
        """
        if x_feat is None:
            x_feat = x_seq[:, -1, :]  # (batch, input_dim) — last timestep

        lstm_out = self.lstm_enc(x_seq)    # (batch, hidden_dim*2)
        cnn_out  = self.cnn_cls(x_feat)   # (batch, num_filters)

        fused  = torch.cat([lstm_out, cnn_out], dim=-1)
        logits = self.head(fused)
        return logits

    def predict_proba(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. For inference use."""
        with torch.no_grad():
            logits = self.forward(x_seq)
            return torch.softmax(logits, dim=-1)

    def predict(self, x_seq: torch.Tensor) -> tuple:
        """
        Returns (predicted_class, confidence_score).
        """
        proba = self.predict_proba(x_seq)
        confidence, predicted = proba.max(dim=-1)
        return predicted, confidence
