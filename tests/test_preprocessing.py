"""
test_preprocessing.py
Unit tests for feature engineering and preprocessing utilities.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.feature_engineering import (
    clean_dataframe, encode_labels, create_sliding_windows, CICIDS_FEATURES
)


class TestCleanDataFrame:
    def test_replaces_inf_with_finite(self):
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf], "b": [2.0, 3.0, 4.0]})
        cleaned = clean_dataframe(df)
        assert not np.any(np.isinf(cleaned["a"].values))

    def test_fills_nan_with_median(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 2.0, 2.0]})
        cleaned = clean_dataframe(df)
        assert not cleaned["a"].isna().any()
        assert cleaned["a"].iloc[1] == pytest.approx(2.0)  # median of [1, 3]


class TestEncodeLabels:
    def test_known_labels_encoded_correctly(self):
        class_names = ["BENIGN", "DDoS", "PortScan"]
        series = pd.Series(["BENIGN", "DDoS", "PortScan", "BENIGN"])
        encoded = encode_labels(series, class_names)
        assert list(encoded) == [0, 1, 2, 0]

    def test_unknown_labels_default_to_zero(self):
        class_names = ["BENIGN", "DDoS"]
        series = pd.Series(["BENIGN", "UNKNOWN_ATTACK"])
        encoded = encode_labels(series, class_names)
        assert encoded[1] == 0

    def test_strips_whitespace(self):
        class_names = ["BENIGN", "DDoS"]
        series = pd.Series([" BENIGN ", " DDoS "])
        encoded = encode_labels(series, class_names)
        assert list(encoded) == [0, 1]


class TestSlidingWindows:
    def test_output_shapes(self):
        X = np.random.randn(100, 78).astype(np.float32)
        y = np.random.randint(0, 15, 100)
        X_w, y_w = create_sliding_windows(X, y, window_size=10)
        assert X_w.shape == (90, 10, 78)
        assert y_w.shape == (90,)

    def test_label_corresponds_to_last_timestep(self):
        X = np.arange(20).reshape(20, 1).astype(np.float32)
        y = np.arange(20)
        X_w, y_w = create_sliding_windows(X, y, window_size=5)
        # First window: X[0:5], label = y[5]
        assert y_w[0] == 5
        assert X_w[0, 0, 0] == pytest.approx(0.0)
        assert X_w[0, 4, 0] == pytest.approx(4.0)

    def test_raises_on_insufficient_data(self):
        X = np.random.randn(5, 78).astype(np.float32)
        y = np.zeros(5, dtype=int)
        with pytest.raises(ValueError):
            create_sliding_windows(X, y, window_size=10)
