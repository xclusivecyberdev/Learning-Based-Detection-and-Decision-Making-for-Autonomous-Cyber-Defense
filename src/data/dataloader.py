"""
dataloader.py
Builds PyTorch DataLoaders from processed splits.
Returns class_weights tensor for use in FocalLoss.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import logging

from src.data.dataset import ThreatDataset

logger = logging.getLogger(__name__)


def compute_class_weights(dataset: ThreatDataset, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency class weighting.
    w_c = total_samples / (num_classes * count_c)
    """
    counts = dataset.class_counts.float()
    counts = counts.clamp(min=1)  # avoid division by zero for missing classes
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights / weights.sum() * num_classes  # normalize
    logger.info(f"Class weights (min={weights.min():.3f}, max={weights.max():.3f})")
    return weights


def build_dataloaders(cfg: dict):
    """
    Build train/val/test DataLoaders and compute class weights.

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    processed_dir = cfg["dataset"]["processed_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_classes = cfg["dataset"]["num_classes"]
    num_workers = 4 if torch.cuda.is_available() else 0

    train_ds = ThreatDataset(processed_dir, split="train")
    val_ds   = ThreatDataset(processed_dir, split="val")
    test_ds  = ThreatDataset(processed_dir, split="test")

    logger.info(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    class_weights = compute_class_weights(train_ds, num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_weights
