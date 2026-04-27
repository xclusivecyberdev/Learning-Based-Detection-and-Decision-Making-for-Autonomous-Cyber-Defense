"""
export_onnx.py
Exports the trained PyTorch model to ONNX format for deployment.

Usage:
    python scripts/export_onnx.py --checkpoint results/checkpoints/best.pt
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.threat_model import ThreatDetectionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/best.pt")
    parser.add_argument("--output", type=str, default="results/threat_model.onnx")
    parser.add_argument("--window_size", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")  # ONNX export on CPU for portability
    model = ThreatDetectionModel(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_lstm_layers"],
        num_filters=cfg["model"]["num_cnn_filters"],
        num_classes=cfg["model"]["num_classes"],
        dropout=0.0,  # disable dropout for export
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Dummy input: (batch=1, window_size, input_dim)
    dummy_input = torch.randn(1, args.window_size, cfg["model"]["input_dim"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=14,
        input_names=["flow_window"],
        output_names=["threat_logits"],
        dynamic_axes={
            "flow_window": {0: "batch_size"},
            "threat_logits": {0: "batch_size"},
        },
    )
    print(f"Model exported to ONNX: {args.output}")


if __name__ == "__main__":
    main()
