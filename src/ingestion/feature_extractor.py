"""
feature_extractor.py
Real-time feature extraction: converts raw log rows → normalised tensor.
"""

import numpy as np
import pandas as pd
import logging
from src.data.feature_engineering import extract_features, load_scaler, create_sliding_windows

logger = logging.getLogger(__name__)


class RealTimeFeatureExtractor:
    """
    Maintains a rolling buffer of recent flows.
    When buffer reaches window_size, yields a normalized feature window.
    """

    def __init__(self, scaler_path: str, window_size: int = 50):
        self.scaler = load_scaler(scaler_path)
        self.window_size = window_size
        self._buffer: list = []  # list of normalised feature vectors (78,)

    def update(self, flow_row: dict) -> np.ndarray | None:
        """
        Add one flow row, return window if buffer is full, else None.

        Args:
            flow_row: dict of raw CICIDS-schema features
        Returns:
            np.ndarray of shape (window_size, 78) or None
        """
        df = pd.DataFrame([flow_row])
        x = extract_features(df, scaler=self.scaler, fit=False,
                              scaler_path="")  # scaler already loaded
        self._buffer.append(x[0])

        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        if len(self._buffer) == self.window_size:
            return np.array(self._buffer, dtype=np.float32)
        return None
