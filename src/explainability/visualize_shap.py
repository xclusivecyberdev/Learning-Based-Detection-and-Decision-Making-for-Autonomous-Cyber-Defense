"""
visualize_shap.py
Renders and saves SHAP visualisation plots.
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def plot_shap_summary(shap_values, X: np.ndarray,
                      feature_names: list, figures_dir: str,
                      class_idx: int = None):
    """
    Beeswarm summary plot of SHAP values.

    Args:
        shap_values : list of (N, window, 78) arrays per class, or single array
        X           : (N, window, 78) input data
        feature_names: list of 78 feature names
        figures_dir : output directory
        class_idx   : if set, plot only for this class; else plot macro mean
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available — skipping summary plot")
        return

    os.makedirs(figures_dir, exist_ok=True)

    # Collapse window dimension: mean over timesteps → (N, 78)
    if isinstance(shap_values, list):
        if class_idx is not None:
            sv = shap_values[class_idx]
        else:
            sv = np.stack(shap_values, axis=0).mean(axis=0)
        sv_2d = sv.mean(axis=1)  # (N, 78)
    else:
        sv_2d = shap_values.mean(axis=1)

    X_2d = X.mean(axis=1)  # (N, 78)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_2d, X_2d, feature_names=feature_names,
                      show=False, max_display=20, plot_type="dot")
    plt.tight_layout()

    tag = f"class_{class_idx}" if class_idx is not None else "macro"
    path = os.path.join(figures_dir, f"shap_summary_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved to {path}")
    return path


def plot_top_features_bar(importance_dict: dict, figures_dir: str,
                          top_n: int = 20):
    """Bar chart of top-N most important features by mean |SHAP|."""
    os.makedirs(figures_dir, exist_ok=True)

    features = list(importance_dict.keys())[:top_n]
    values   = list(importance_dict.values())[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(features)), values[::-1], color="#185FA5", alpha=0.85)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_n} Features by SHAP Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(figures_dir, "shap_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP feature importance bar chart saved to {path}")
    return path
