"""
shap_explainer.py
SHAP-based feature importance for the threat detection model.

Provides per-prediction explanations showing which network flow
features drove each classification decision.

Reference:
    Lundberg, S. M., & Lee, S. I. (2017). A unified approach to
    interpreting model predictions. NeurIPS 2017.
    https://arxiv.org/abs/1705.07874

Research significance:
    Explainability is a hard requirement for deploying autonomous
    agents in operational security environments. SHAP values provide
    a mathematically grounded attribution method (Shapley values from
    cooperative game theory) that satisfies desirable properties:
    local accuracy, missingness, and consistency.
"""

import numpy as np
import torch
import logging
import os

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")


class ThreatSHAPExplainer:
    """
    Wraps the ThreatDetectionModel for SHAP DeepExplainer.

    Uses a background dataset (random sample of benign traffic) to
    compute expected SHAP values across the feature space.
    """

    def __init__(self, model, background_data: np.ndarray,
                 feature_names: list, device: torch.device):
        """
        Args:
            model           : trained ThreatDetectionModel
            background_data : (N, window_size, 78) numpy array of background samples
            feature_names   : list of 78 feature name strings
            device          : torch device
        """
        if not SHAP_AVAILABLE:
            raise ImportError("Install shap: pip install shap")

        self.model = model.eval()
        self.device = device
        self.feature_names = feature_names

        # Wrap model forward for SHAP (takes sequence, returns logits)
        def model_forward(x):
            x_t = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                logits = model(x_t)
            return logits.cpu().numpy()

        bg_tensor = torch.FloatTensor(background_data).to(device)
        self.explainer = shap.DeepExplainer(model, bg_tensor)
        logger.info(f"SHAP explainer initialised with {len(background_data)} background samples")

    def explain_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a batch of windows.

        Args:
            X : (batch, window_size, 78)
        Returns:
            shap_values : (batch, window_size, 78, num_classes)
        """
        X_t = torch.FloatTensor(X).to(self.device)
        shap_values = self.explainer.shap_values(X_t)
        return shap_values

    def get_feature_importance(self, X: np.ndarray) -> dict:
        """
        Compute mean absolute SHAP value per feature (across timesteps and classes).
        Returns a dict sorted by importance descending.
        """
        shap_vals = self.explain_batch(X)  # list of (batch, window, 78) per class
        # Stack: (num_classes, batch, window, 78)
        stacked = np.stack(shap_vals, axis=0)
        # Mean over classes, batch, timesteps → (78,)
        importance = np.abs(stacked).mean(axis=(0, 1, 2))
        return dict(sorted(
            zip(self.feature_names, importance.tolist()),
            key=lambda x: x[1], reverse=True
        ))
