"""
test_model.py
Unit tests for the detection model components.
"""

import pytest
import torch
from src.models.lstm_encoder import BiLSTMEncoder
from src.models.cnn_classifier import CNNClassifier
from src.models.threat_model import ThreatDetectionModel
from src.models.focal_loss import FocalLoss


class TestBiLSTMEncoder:
    def test_output_shape(self, sample_batch, device):
        encoder = BiLSTMEncoder(input_dim=78, hidden_dim=256, num_layers=2).to(device)
        out = encoder(sample_batch.to(device))
        assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"

    def test_output_dim_property(self):
        encoder = BiLSTMEncoder(hidden_dim=128)
        assert encoder.output_dim == 256

    def test_no_nan_in_output(self, sample_batch, device):
        encoder = BiLSTMEncoder().to(device)
        out = encoder(sample_batch.to(device))
        assert not torch.isnan(out).any(), "NaN values in LSTM output"


class TestCNNClassifier:
    def test_output_shape(self, sample_batch, device):
        cnn = CNNClassifier(input_dim=78, num_filters=128).to(device)
        x = sample_batch[:, -1, :].to(device)  # last frame
        out = cnn(x)
        assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"

    def test_output_dim_property(self):
        cnn = CNNClassifier(num_filters=64)
        assert cnn.output_dim == 64


class TestThreatDetectionModel:
    def test_logits_shape(self, sample_batch, device):
        model = ThreatDetectionModel(num_classes=15).to(device)
        logits = model(sample_batch.to(device))
        assert logits.shape == (4, 15), f"Expected (4, 15), got {logits.shape}"

    def test_predict_proba_sums_to_one(self, sample_batch, device):
        model = ThreatDetectionModel(num_classes=15).to(device)
        proba = model.predict_proba(sample_batch.to(device))
        sums = proba.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4, device=device), atol=1e-5), \
            "Probabilities do not sum to 1"

    def test_predict_returns_valid_class(self, sample_batch, device):
        model = ThreatDetectionModel(num_classes=15).to(device)
        preds, confs = model.predict(sample_batch.to(device))
        assert preds.shape == (4,)
        assert all(0 <= p.item() < 15 for p in preds)
        assert all(0.0 <= c.item() <= 1.0 for c in confs)


class TestFocalLoss:
    def test_loss_is_scalar(self, device):
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(8, 15)
        targets = torch.randint(0, 15, (8,))
        loss = loss_fn(logits, targets)
        assert loss.shape == torch.Size([]), "Loss should be scalar"

    def test_loss_nonnegative(self, device):
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(8, 15)
        targets = torch.randint(0, 15, (8,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_gamma_zero_matches_cross_entropy(self, device):
        """When gamma=0, focal loss should equal cross-entropy."""
        import torch.nn.functional as F
        logits  = torch.randn(8, 15)
        targets = torch.randint(0, 15, (8,))
        focal = FocalLoss(gamma=0.0)(logits, targets)
        ce    = F.cross_entropy(logits, targets)
        assert abs(focal.item() - ce.item()) < 1e-5
