"""
callbacks.py
Reusable training callbacks: checkpoint saver, LR logger, curve plotter.
"""

import os
import json
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_curves(history: dict, figures_dir: str):
    """Plot and save loss and accuracy curves from training history dict."""
    os.makedirs(figures_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#185FA5")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#E24B4A")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#1D9E75")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(figures_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved to {path}")
    return path


def plot_confusion_matrix(cm: list, class_names: list, figures_dir: str):
    """Plot and save a normalised confusion matrix heatmap."""
    import numpy as np
    import seaborn as sns

    os.makedirs(figures_dir, exist_ok=True)
    cm_arr = np.array(cm)
    # Row-normalise
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.3,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalised Confusion Matrix")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {path}")
    return path


def plot_roc_curves(y_true, y_proba, class_names: list, figures_dir: str):
    """Plot one-vs-rest ROC curves for all classes."""
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    os.makedirs(figures_dir, exist_ok=True)
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{name} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved to {path}")
    return path
