"""
evaluate.py
Full evaluation pipeline: detection model + RL agent + baseline comparison.

Produces:
    - Classification report (per-class F1, precision, recall)
    - Confusion matrix plot
    - ROC curves
    - Ablation study: LSTM-only vs CNN-only vs Hybrid
    - Agent comparison: DQN vs Rule-based vs Random
    - SHAP feature importance (optional, requires shap)
    - All results saved to results/

Usage:
    python scripts/evaluate.py --config configs/train_config.yaml
    python scripts/evaluate.py --config configs/train_config.yaml --ablation
    python scripts/evaluate.py --config configs/train_config.yaml --shap
"""

import argparse
import os
import sys
import json
import yaml
import torch
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import build_dataloaders
from src.models.threat_model import ThreatDetectionModel
from src.models.lstm_encoder import BiLSTMEncoder
from src.models.cnn_classifier import CNNClassifier
from src.training.metrics import compute_metrics, save_metrics
from src.training.callbacks import plot_confusion_matrix, plot_roc_curves, plot_training_curves
from src.agent.agent_core import AgenticDefenseSystem
from src.agent.policy import RuleBasedPolicy
from src.agent.rl_environment import CyberDefenseEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(cfg, ckpt_path, device):
    model = ThreatDetectionModel(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_lstm_layers"],
        num_filters=cfg["model"]["num_cnn_filters"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_ablation(cfg, train_loader, val_loader, test_loader, device, class_names, figures_dir):
    """
    Ablation study: compare LSTM-only, CNN-only, and Hybrid models.
    Returns dict of results for each variant.
    """
    logger.info("\n=== ABLATION STUDY ===")
    results = {}

    # We evaluate all three variants using their saved checkpoints
    # If not available, we report that training is needed
    variants = {
        "lstm_only": os.path.join(cfg["paths"]["checkpoint_dir"], "lstm_only_best.pt"),
        "cnn_only":  os.path.join(cfg["paths"]["checkpoint_dir"], "cnn_only_best.pt"),
        "hybrid":    os.path.join(cfg["paths"]["checkpoint_dir"], "best.pt"),
    }

    for name, ckpt_path in variants.items():
        if not os.path.exists(ckpt_path):
            logger.warning(f"Ablation: checkpoint for '{name}' not found at {ckpt_path}. "
                           f"Train this variant first.")
            results[name] = {"status": "checkpoint_not_found", "path": ckpt_path}
            continue

        model = load_model(cfg, ckpt_path, device)
        metrics = compute_metrics(model, test_loader, device,
                                  cfg["model"]["num_classes"], class_names)
        results[name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc":  metrics["roc_auc"],
            "fpr":      metrics["fpr"],
        }
        logger.info(f"  {name:12s} | acc={metrics['accuracy']:.4f} | "
                    f"F1={metrics['macro_f1']:.4f} | AUC={metrics['roc_auc']:.4f}")

    return results


def run_agent_comparison(detection_model, test_loader, full_cfg, device):
    """
    Compare DQN agent vs Rule-based baseline vs Random policy.
    Returns dict of agent metrics for each policy.
    """
    logger.info("\n=== AGENT POLICY COMPARISON ===")

    # Collect (prob_vector, true_label) events from test set
    detection_model.eval()
    events = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            probs = detection_model.predict_proba(X).cpu().numpy()
            for p, label in zip(probs, y.numpy()):
                events.append((p, int(label)))

    env = CyberDefenseEnv(full_cfg["rewards"])
    results = {}

    # ── DQN Agent ────────────────────────────────────────────────────────
    rl_ckpt = full_cfg["paths"].get("rl_checkpoint",
                                    "results/checkpoints/rl_agent.pt")
    if os.path.exists(rl_ckpt):
        from src.agent.dqn_agent import DQNAgent
        dqn = DQNAgent(full_cfg, device)
        dqn.load(rl_ckpt)
        results["dqn"] = _eval_policy_on_events(dqn, env, events, mode="rl")
        logger.info(f"  DQN       | F1={results['dqn']['f1']:.4f} | "
                    f"FPR={results['dqn']['fpr']:.4f} | "
                    f"MTTR={results['dqn']['mean_response_time_ms']:.2f}ms")
    else:
        logger.warning(f"DQN checkpoint not found at {rl_ckpt}. Run RL training first.")
        results["dqn"] = {"status": "not_trained"}

    # ── Rule-Based Baseline ───────────────────────────────────────────────
    rule_policy = RuleBasedPolicy(full_cfg["policy"])
    results["rule_based"] = _eval_policy_on_events(rule_policy, env, events, mode="rule")
    logger.info(f"  Rule-based | F1={results['rule_based']['f1']:.4f} | "
                f"FPR={results['rule_based']['fpr']:.4f} | "
                f"MTTR={results['rule_based']['mean_response_time_ms']:.2f}ms")

    # ── Random Policy ────────────────────────────────────────────────────
    import random
    class RandomPolicy:
        def select_action(self, _):
            return random.randint(0, 3)
    random_policy = RandomPolicy()
    results["random"] = _eval_policy_on_events(random_policy, env, events, mode="rule")
    logger.info(f"  Random     | F1={results['random']['f1']:.4f} | "
                f"FPR={results['random']['fpr']:.4f}")

    return results


def _eval_policy_on_events(policy, env, events, mode="rule"):
    import time
    tp = fp = fn = tn = 0
    response_times = []
    total_reward = 0.0

    for prob_vec, true_label in events:
        t0 = time.perf_counter()
        if mode == "rl":
            state = env._make_state(prob_vec)
            action = policy.select_action(state, training=False)
        else:
            action = policy.select_action(prob_vec)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        response_times.append(elapsed_ms)

        reward, info = env._compute_reward(action, true_label, prob_vec)
        total_reward += reward
        outcome = info["outcome"]
        if outcome == "TP":    tp += 1
        elif "FP" in outcome:  fp += 1
        elif outcome == "FN":  fn += 1
        elif outcome == "TN":  tn += 1

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    fpr       = fp / max(fp + tn, 1)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "total_reward": round(total_reward, 3),
        "mean_response_time_ms": round(float(np.mean(response_times)), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="configs/train_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/agent_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--shap",     action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.agent_config) as f:
        agent_cfg = yaml.safe_load(f)
    full_cfg = {**cfg, **agent_cfg}
    full_cfg["class_names"] = cfg["dataset"]["class_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = cfg["dataset"]["class_names"]
    figures_dir = cfg["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)

    # ── Load Data ─────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg)

    # ── Load Best Detection Model ─────────────────────────────────────────
    ckpt_path = args.checkpoint or os.path.join(cfg["paths"]["checkpoint_dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        logger.error(f"No checkpoint found at {ckpt_path}. Train first: python train.py")
        sys.exit(1)

    model = load_model(cfg, ckpt_path, device)
    logger.info(f"Loaded detection model from {ckpt_path}")

    # ── Detection Model Evaluation ─────────────────────────────────────────
    logger.info("\n=== DETECTION MODEL EVALUATION ===")
    metrics = compute_metrics(model, test_loader, device,
                              cfg["model"]["num_classes"], class_names)

    print("\n" + "=" * 65)
    print("DETECTION MODEL — TEST SET RESULTS")
    print("=" * 65)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Mean FPR  : {metrics['fpr']:.4f}")
    print("=" * 65)
    print(metrics["classification_report"])

    # Save metrics
    save_metrics(metrics, os.path.join(cfg["paths"]["results_dir"], "metrics.json"))

    # Plots
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, figures_dir)
    logger.info("Confusion matrix saved.")

    # Training curves (from saved history if available)
    history_path = os.path.join(cfg["paths"]["log_dir"], "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, figures_dir)

    # ── Ablation Study ────────────────────────────────────────────────────
    if args.ablation:
        ablation_results = run_ablation(cfg, train_loader, val_loader,
                                        test_loader, device, class_names, figures_dir)
        ablation_path = os.path.join(cfg["paths"]["results_dir"], "ablation.json")
        with open(ablation_path, "w") as f:
            json.dump(ablation_results, f, indent=2)
        logger.info(f"Ablation results saved to {ablation_path}")

    # ── Agent Comparison ──────────────────────────────────────────────────
    agent_results = run_agent_comparison(model, test_loader, full_cfg, device)
    agent_path = os.path.join(cfg["paths"]["results_dir"], "agent_comparison.json")
    with open(agent_path, "w") as f:
        json.dump(agent_results, f, indent=2)
    logger.info(f"Agent comparison saved to {agent_path}")

    print("\n=== AGENT POLICY COMPARISON ===")
    for policy_name, res in agent_results.items():
        if "status" in res:
            print(f"  {policy_name:12s} | {res['status']}")
        else:
            print(f"  {policy_name:12s} | F1={res['f1']:.4f} | "
                  f"FPR={res['fpr']:.4f} | MTTR={res['mean_response_time_ms']:.2f}ms")

    # ── SHAP Explainability ───────────────────────────────────────────────
    if args.shap:
        try:
            from src.explainability.shap_explainer import ThreatSHAPExplainer
            from src.explainability.visualize_shap import plot_top_features_bar
            from src.data.feature_engineering import CICIDS_FEATURES

            logger.info("\n=== SHAP EXPLAINABILITY ===")
            # Sample 100 background examples
            bg_batch = next(iter(test_loader))
            bg_X = bg_batch[0][:100].numpy()

            explainer = ThreatSHAPExplainer(model, bg_X, CICIDS_FEATURES, device)
            sample_X = bg_batch[0][:50].numpy()
            importance = explainer.get_feature_importance(sample_X)
            plot_top_features_bar(importance, figures_dir)
            importance_path = os.path.join(cfg["paths"]["results_dir"], "shap_importance.json")
            with open(importance_path, "w") as f:
                json.dump(importance, f, indent=2)
            logger.info(f"SHAP importance saved to {importance_path}")
        except Exception as e:
            logger.error(f"SHAP failed: {e}. Install shap: pip install shap")

    logger.info("\n✓ Evaluation complete. Check results/ directory.")


if __name__ == "__main__":
    main()
