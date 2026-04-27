"""
stream_simulator.py
Simulates a live event stream from a preprocessed CSV file.
Used for testing the agent pipeline without a real network.
"""

import numpy as np
import pandas as pd
import time
import logging
from src.data.feature_engineering import extract_features, create_sliding_windows, load_scaler, CICIDS_FEATURES

logger = logging.getLogger(__name__)


class StreamSimulator:
    """
    Replays preprocessed CICIDS2017 test data as a simulated live stream.
    Yields (flow_window: np.ndarray[50, 78], true_label: int) tuples.
    """

    def __init__(self, data_path: str, scaler_path: str, window_size: int = 50,
                 speed: float = 0.0, class_names: list = None):
        """
        Args:
            data_path   : Path to a CSV with CICIDS2017 schema
            scaler_path : Path to fitted MinMaxScaler pkl
            window_size : Sequence length for LSTM input
            speed       : Seconds to wait between events (0 = as fast as possible)
            class_names : List of class label strings
        """
        self.window_size = window_size
        self.speed = speed
        self.class_names = class_names or []

        logger.info(f"Loading simulation data from {data_path}")
        df = pd.read_csv(data_path, low_memory=False)

        label_col = "Label" if "Label" in df.columns else df.columns[-1]
        y_raw = df[label_col].copy()

        # Build label encoder from class_names
        label_map = {name: idx for idx, name in enumerate(self.class_names)}
        y = y_raw.str.strip().map(label_map).fillna(0).astype(int).values

        scaler = load_scaler(scaler_path)
        X = extract_features(df, scaler=scaler, fit=False, scaler_path=scaler_path)
        X_windows, y_windows = create_sliding_windows(X, y, window_size)

        self.X_windows = X_windows  # (N, window_size, 78)
        self.y_windows = y_windows  # (N,)
        self.n_events = len(y_windows)
        logger.info(f"Stream simulator ready: {self.n_events} events")

    def stream(self):
        """Yield (flow_window, true_label) pairs indefinitely (loops once)."""
        for i in range(self.n_events):
            if self.speed > 0:
                time.sleep(self.speed)
            yield self.X_windows[i], int(self.y_windows[i])

    def get_events_for_rl(self) -> list:
        """Return events as list of (prob_placeholder, true_label) for RL env init."""
        # Returns raw windows; the agent_core will call perceive() to get proba
        return list(zip(self.X_windows, self.y_windows))
