"""
run_agent.py
Entry point for the agentic cyber defense system.

Usage:
    # Simulation mode (replay test CSV — no real network needed)
    python run_agent.py --mode simulate --data data/sample/sample_traffic.csv

    # Live mode (monitor a directory for new log files)
    python run_agent.py --mode live --log_dir /var/log/network/

    # Train RL agent (runs after detection model is trained)
    python run_agent.py --mode train_rl

    # Compare DQN vs rule-based (evaluation mode)
    python run_agent.py --mode compare
"""

import argparse
import os
import sys
import yaml
import json
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from src.models.threat_model import ThreatDetectionModel
from src.agent.agent_core import AgenticDefenseSystem
from src.data.dataloader import build_dataloaders


def load_models(cfg, agent_cfg, device):
    model = ThreatDetectionModel(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_lstm_layers"],
        num_filters=cfg["model"]["num_cnn_filters"],
        num_classes=cfg["model"]["num_classes"],
        dropout=0.0,
    ).to(device)
    ckpt_path = os.path.join(cfg["paths"]["checkpoint_dir"], "best.pt")
    if not os.path.exists(ckpt_path):
        logger.error(f"Detection model checkpoint not found: {ckpt_path}")
        logger.error("Train first: python train.py")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


def mode_simulate(detection_model, full_cfg, device, data_path, policy_mode):
    """Replay a CSV through the full Perceive→Act pipeline."""
    from src.ingestion.stream_simulator import StreamSimulator

    scaler_path = os.path.join(full_cfg["dataset"]["processed_dir"], "scaler.pkl")
    if not os.path.exists(scaler_path):
        logger.error(f"Scaler not found at {scaler_path}. Run preprocessing first.")
        sys.exit(1)

    simulator = StreamSimulator(
        data_path=data_path,
        scaler_path=scaler_path,
        window_size=full_cfg["dataset"]["window_size"],
        speed=0.0,
        class_names=full_cfg["dataset"]["class_names"],
    )

    agent = AgenticDefenseSystem(detection_model, full_cfg, device, mode=policy_mode)

    print(f"\n[AGENT] Starting simulation | policy={policy_mode} | events={simulator.n_events}")
    print("-" * 65)

    for i, (window, true_label) in enumerate(simulator.stream()):
        decision = agent.process_event(window, true_label=true_label)

        if decision["action"] != "IGNORE":
            print(
                f"[{i+1:06d}] {decision['action']:8s} | "
                f"{decision['predicted_threat'][:25]:25s} | "
                f"conf={decision['confidence']:.3f} | "
                f"{decision['outcome']} | {decision['response_time_ms']:.1f}ms"
            )

        if (i + 1) % 1000 == 0:
            summary = agent.get_summary()
            print(f"\n--- Summary @ {i+1} events ---")
            print(f"  F1={summary['f1']:.4f} | FPR={summary['fpr']:.4f} | "
                  f"MTTR={summary['mean_response_time_ms']:.2f}ms | "
                  f"Reward={summary['total_reward']:.1f}")
            print()

    summary = agent.get_summary()
    print("\n" + "=" * 65)
    print("SIMULATION COMPLETE")
    print("=" * 65)
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")
    print("=" * 65)

    results_path = os.path.join(full_cfg.get("results_dir", "results"),
                                 f"simulation_{policy_mode}.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary saved to {results_path}")


def mode_train_rl(detection_model, full_cfg, device):
    """Train the DQN agent using the trained detection model."""
    from src.agent.rl_trainer import train_rl_agent
    train_loader, val_loader, _, _ = build_dataloaders(full_cfg)
    logger.info("Starting RL agent training...")
    agent = train_rl_agent(detection_model, train_loader, val_loader, full_cfg, device)
    logger.info("RL training complete.")


def mode_live(detection_model, full_cfg, device, log_dir):
    """Monitor a log directory and process events in real time."""
    from src.ingestion.log_reader import LogReader
    from src.ingestion.feature_extractor import RealTimeFeatureExtractor

    scaler_path = os.path.join(full_cfg["dataset"]["processed_dir"], "scaler.pkl")
    extractor = RealTimeFeatureExtractor(scaler_path, full_cfg["dataset"]["window_size"])
    reader    = LogReader(log_dir)
    agent     = AgenticDefenseSystem(detection_model, full_cfg, device, mode="rl")

    print(f"[AGENT] Live mode — monitoring {log_dir}")
    log_files = reader.list_log_files()
    if not log_files:
        logger.error(f"No CSV log files found in {log_dir}")
        sys.exit(1)

    for df_chunk in reader.tail_csv(log_files[-1]):
        for _, row in df_chunk.iterrows():
            window = extractor.update(row.to_dict())
            if window is not None:
                decision = agent.process_event(window)
                if decision["action"] != "IGNORE":
                    print(f"[LIVE] {decision['action']} | {decision['predicted_threat']} | "
                          f"conf={decision['confidence']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, default="configs/train_config.yaml")
    parser.add_argument("--agent_config", type=str, default="configs/agent_config.yaml")
    parser.add_argument("--mode",  type=str, default="simulate",
                        choices=["simulate", "live", "train_rl", "compare"])
    parser.add_argument("--data",    type=str, default="data/sample/sample_traffic.csv")
    parser.add_argument("--log_dir", type=str, default="/var/log/network/")
    parser.add_argument("--policy",  type=str, default="rl", choices=["rl", "rule"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.agent_config) as f:
        agent_cfg = yaml.safe_load(f)

    full_cfg = {**cfg, **agent_cfg}
    full_cfg["class_names"] = cfg["dataset"]["class_names"]
    full_cfg["results_dir"] = cfg["paths"]["results_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Mode: {args.mode}")

    detection_model = load_models(cfg, agent_cfg, device)

    if args.mode == "simulate":
        mode_simulate(detection_model, full_cfg, device, args.data, args.policy)
    elif args.mode == "train_rl":
        mode_train_rl(detection_model, full_cfg, device)
    elif args.mode == "live":
        mode_live(detection_model, full_cfg, device, args.log_dir)
    elif args.mode == "compare":
        # Run simulation with both policies and compare
        logger.info("Running DQN policy...")
        mode_simulate(detection_model, full_cfg, device, args.data, "rl")
        logger.info("Running rule-based baseline...")
        mode_simulate(detection_model, full_cfg, device, args.data, "rule")


if __name__ == "__main__":
    main()
