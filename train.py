"""
train.py
Entry point: trains the BiLSTM+CNN threat detection model.

Usage:
    python train.py
    python train.py --config configs/train_config.yaml
    python train.py --config configs/train_config.yaml --resume results/checkpoints/latest.pt
    python train.py --variant lstm_only
    python train.py --variant cnn_only
"""

import argparse
import yaml
import os
import sys
import torch
import random
import numpy as np
import json

from src.data.dataloader import build_dataloaders
from src.models.threat_model import ThreatDetectionModel
from src.models.focal_loss import FocalLoss
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, save_metrics
from src.training.callbacks import plot_training_curves


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default="configs/train_config.yaml")
    parser.add_argument("--resume",  type=str, default=None)
    parser.add_argument("--variant", type=str, default="hybrid",
                        choices=["hybrid", "lstm_only", "cnn_only"],
                        help="Architecture variant for ablation study")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device      : {device}")
    print(f"[INFO] Variant     : {args.variant}")
    print(f"[INFO] Config      : {args.config}")

    # ── Data ─────────────────────────────────────────────────────────────
    print("[INFO] Building dataloaders...")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)

    # ── Model (ablation support) ──────────────────────────────────────────
    # For ablation: lstm_only disables CNN branch (num_filters=0),
    # cnn_only disables LSTM branch (hidden_dim=0).
    # Both variants still use ThreatDetectionModel but with zeroed components.
    model_cfg = dict(cfg["model"])
    if args.variant == "lstm_only":
        model_cfg["num_cnn_filters"] = 0   # CNN disabled internally
    elif args.variant == "cnn_only":
        model_cfg["hidden_dim"] = 0        # LSTM disabled internally

    model = ThreatDetectionModel(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_lstm_layers"],
        num_filters=model_cfg["num_cnn_filters"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters  : {total_params:,}")

    # ── Loss & Optimizer ──────────────────────────────────────────────────
    criterion = FocalLoss(
        gamma=cfg["training"]["focal_gamma"],
        alpha=class_weights.to(device) if cfg["training"]["use_class_weights"] else None,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=cfg["training"]["min_lr"],
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[INFO] Resumed from epoch {start_epoch}")

    # Override checkpoint dir for ablation variants
    if args.variant != "hybrid":
        cfg["paths"]["checkpoint_dir"] = os.path.join(
            cfg["paths"]["checkpoint_dir"], args.variant
        )
        # Rename best checkpoint to include variant
        # (evaluate.py looks for {variant}_best.pt in base checkpoint dir)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, criterion, optimizer, scheduler, device, cfg, start_epoch)
    print("[INFO] Starting training...")
    trainer.fit(train_loader, val_loader)

    # ── Final Test Evaluation ─────────────────────────────────────────────
    best_ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "best.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    class_names = cfg["dataset"]["class_names"]
    test_metrics = compute_metrics(model, test_loader, device,
                                   cfg["model"]["num_classes"], class_names)

    print("\n" + "=" * 65)
    print(f"FINAL TEST RESULTS — variant={args.variant}")
    print("=" * 65)
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1  : {test_metrics['macro_f1']:.4f}")
    print(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f"  Mean FPR  : {test_metrics['fpr']:.4f}")
    print("=" * 65)
    print(test_metrics["classification_report"])

    # Save metrics per variant
    metrics_filename = f"metrics_{args.variant}.json" if args.variant != "hybrid" else "metrics.json"
    save_metrics(test_metrics, os.path.join(cfg["paths"]["results_dir"], metrics_filename))

    # Plot training curves
    history_path = os.path.join(cfg["paths"]["log_dir"], "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, cfg["paths"]["figures_dir"])


if __name__ == "__main__":
    main()
