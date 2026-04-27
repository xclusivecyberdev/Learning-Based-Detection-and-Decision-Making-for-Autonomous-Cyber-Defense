"""
lstm_encoder.py
Bidirectional LSTM encoder for sequential network log analysis.

Research rationale:
    BiLSTMs capture both forward and backward temporal dependencies in
    network flow sequences. This is important for cyber threats like
    slow-rate DoS attacks, where the threat pattern only becomes visible
    when looking at the flow history in both directions.
"""

import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    """
    2-layer Bidirectional LSTM encoder.

    Input : (batch, seq_len, input_dim)
    Output: (batch, hidden_dim * 2)  — final hidden state concatenation
    """

    def __init__(self, input_dim: int = 78, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Layer norm on output helps stabilize training with focal loss
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            out: (batch, hidden_dim * 2) — last timestep representation
        """
        # lstm_out: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Take last timestep: (batch, hidden_dim * 2)
        out = lstm_out[:, -1, :]
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * 2
