"""
dataset.py
PyTorch Dataset class for the processed CICIDS2017 threat detection data.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class ThreatDataset(Dataset):
    """
    Loads preprocessed (windowed) tensors from disk.
    Expected files in processed_dir:
        X_train.npy, y_train.npy
        X_val.npy,   y_val.npy
        X_test.npy,  y_test.npy
    """

    def __init__(self, processed_dir: str, split: str = "train"):
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        X_path = os.path.join(processed_dir, f"X_{split}.npy")
        y_path = os.path.join(processed_dir, f"y_{split}.npy")

        if not os.path.exists(X_path):
            raise FileNotFoundError(
                f"{X_path} not found. Run: python scripts/preprocess.py"
            )

        self.X = torch.from_numpy(np.load(X_path)).float()   # (N, window, features)
        self.y = torch.from_numpy(np.load(y_path)).long()    # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def num_classes(self):
        return int(self.y.max().item()) + 1

    @property
    def class_counts(self):
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        for label in self.y:
            counts[label] += 1
        return counts


class StreamDataset(Dataset):
    """
    In-memory dataset for real-time simulation.
    Accepts numpy arrays directly (no disk I/O).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
