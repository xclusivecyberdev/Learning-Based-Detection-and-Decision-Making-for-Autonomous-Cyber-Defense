"""
feature_engineering.py
Extracts and normalizes the 78 flow-level features used by CICIDS2017.
Handles missing values, infinite values, and per-feature min-max scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import logging

logger = logging.getLogger(__name__)

# The 78 features used from CICIDS2017 after dropping label & ID columns
CICIDS_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN, then fill NaN with column median."""
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)
    return df


def encode_labels(series: pd.Series, class_names: list) -> np.ndarray:
    """Map string labels to integer indices."""
    label_map = {name: idx for idx, name in enumerate(class_names)}
    # Strip leading/trailing whitespace (CICIDS CSVs often have spaces)
    mapped = series.str.strip().map(label_map)
    unknown = mapped.isna().sum()
    if unknown > 0:
        logger.warning(f"{unknown} unknown labels found — mapping to 0 (BENIGN)")
        mapped = mapped.fillna(0)
    return mapped.astype(int).values


def fit_scaler(X: np.ndarray, scaler_path: str) -> MinMaxScaler:
    """Fit a MinMaxScaler on X and save to disk."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler fitted and saved to {scaler_path}")
    return scaler


def load_scaler(scaler_path: str) -> MinMaxScaler:
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run preprocessing first.")
    return joblib.load(scaler_path)


def extract_features(df: pd.DataFrame, scaler: MinMaxScaler = None,
                     fit: bool = False, scaler_path: str = "data/processed/scaler.pkl") -> np.ndarray:
    """
    Extract and scale the 78 CICIDS features from a DataFrame.
    Returns ndarray of shape (N, 78).
    """
    available = [f for f in CICIDS_FEATURES if f in df.columns]
    missing = [f for f in CICIDS_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"{len(missing)} features missing from DataFrame, filling with 0: {missing[:5]}...")
        for m in missing:
            df[m] = 0.0

    df = clean_dataframe(df)
    X = df[CICIDS_FEATURES].values.astype(np.float32)

    if fit:
        scaler = fit_scaler(X, scaler_path)
    elif scaler is None:
        scaler = load_scaler(scaler_path)

    X = scaler.transform(X)
    return X.astype(np.float32)


def create_sliding_windows(X: np.ndarray, y: np.ndarray,
                            window_size: int = 50) -> tuple:
    """
    Create sliding windows for LSTM sequential input.
    X: (N, features) -> X_windows: (N-window_size, window_size, features)
    y: (N,)          -> y_windows: (N-window_size,)  [label of last step]
    """
    if len(X) <= window_size:
        raise ValueError(f"Not enough samples ({len(X)}) for window_size={window_size}")

    X_windows, y_windows = [], []
    for i in range(window_size, len(X)):
        X_windows.append(X[i - window_size:i])
        y_windows.append(y[i])

    return np.array(X_windows, dtype=np.float32), np.array(y_windows, dtype=np.int64)
