"""
cnn_classifier.py
1D-CNN for spatial pattern recognition in network traffic features.

Research rationale:
    While the LSTM captures temporal ordering, the CNN extracts local
    feature correlations (e.g., the co-occurrence of high packet rate +
    high flag count) that are independent of sequence position.
    Combining both addresses different aspects of the detection problem.

    Reference: LeCun et al. (1998); Kim (2014) for text CNNs adapted here
    to fixed-length feature vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """
    3-layer 1D Convolutional network with global average pooling.

    Input : (batch, input_dim)        — flat feature vector per flow
    Output: (batch, num_filters)
    """

    def __init__(self, input_dim: int = 78, num_filters: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim

        # Three conv layers with increasing filter counts
        self.conv1 = nn.Conv1d(1, num_filters,       kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters,  kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.bn3 = nn.BatchNorm1d(num_filters)

        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pool → (batch, filters, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            out: (batch, num_filters)
        """
        # Treat features as a 1D sequence: (batch, 1, input_dim)
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))   # (batch, num_filters, input_dim)
        x = F.relu(self.bn2(self.conv2(x)))   # (batch, num_filters*2, input_dim)
        x = F.relu(self.bn3(self.conv3(x)))   # (batch, num_filters, input_dim)

        x = self.pool(x).squeeze(-1)          # (batch, num_filters)
        x = self.dropout(x)
        return x

    @property
    def output_dim(self) -> int:
        return self.conv3.out_channels
