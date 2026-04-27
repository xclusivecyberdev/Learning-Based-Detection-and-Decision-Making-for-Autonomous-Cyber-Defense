"""
metrics.py
Evaluation metrics for the threat detection model.
Computes accuracy, per-class F1, macro F1, ROC-AUC, and FPR.
"""

import torch
import numpy as np
import json
import os
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_metrics(model, loader: DataLoader, device: torch.device,
                    num_classes: int, class_names: list = None) -> dict:
    """
    Run model on loader, collect predictions and targets, compute full metrics.

    Returns dict with:
        accuracy, macro_f1, per_class_f1, roc_auc, fpr, confusion_matrix,
        classification_report
    """
    model.eval()
    all_preds, all_targets, all_proba = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())
            all_proba.append(proba)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_proba = np.concatenate(all_proba)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred)

    # False Positive Rate: FP / (FP + TN) averaged over classes
    fpr_per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr_c = fp / (fp + tn + 1e-8)
        fpr_per_class.append(fpr_c)
    mean_fpr = float(np.mean(fpr_per_class))

    # ROC-AUC (one-vs-rest, macro average)
    try:
        roc_auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro"
        )
    except ValueError as e:
        logger.warning(f"ROC-AUC could not be computed: {e}")
        roc_auc = 0.0

    report = classification_report(
        y_true, y_pred,
        target_names=class_names if class_names else None,
        zero_division=0
    )

    logger.info(f"\n{report}")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "roc_auc": float(roc_auc),
        "fpr": mean_fpr,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # classification_report string is not JSON-serialisable as-is; keep it
    saveable = {k: v for k, v in metrics.items() if k != "classification_report"}
    saveable["classification_report_text"] = metrics.get("classification_report", "")
    with open(path, "w") as f:
        json.dump(saveable, f, indent=2)
    logger.info(f"Metrics saved to {path}")
