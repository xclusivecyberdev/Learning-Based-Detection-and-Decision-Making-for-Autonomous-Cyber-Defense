"""
agent_core.py
Agentic decision loop: Perceive → Reason → Act → Reflect.

This module coordinates the detection model and the RL policy into a
unified agent that processes events in real time or simulation mode.

The agent follows the Perceive-Reason-Act-Reflect (PRAR) cycle described in:
    Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
    (4th ed.). Chapter 2: Intelligent Agents.

Key design decisions:
  1. The detection model (BiLSTM+CNN) handles PERCEPTION — converting raw
     network flows into threat probability vectors.
  2. The DQN policy handles REASONING+ACTION — selecting the optimal response
     given the threat state and learned experience.
  3. The incident logger handles REFLECTION — storing decisions for future
     policy improvement and explainability.
"""

import time
import logging
import numpy as np
import torch
from typing import Optional

from src.agent.dqn_agent import DQNAgent, ACTION_NAMES
from src.agent.rl_environment import CyberDefenseEnv, SEVERITY
from src.agent.policy import RuleBasedPolicy
from src.agent.memory import AgentMemory
from src.agent.incident_logger import IncidentLogger
from src.agent.explainer import ThreatExplainer

logger = logging.getLogger(__name__)


class AgenticDefenseSystem:
    """
    Unified agentic cyber defense system.

    Modes:
        'rl'   — use learned DQN policy (primary research mode)
        'rule' — use rule-based policy (baseline for comparison)

    The two modes enable the ablation study comparing learned vs.
    rule-based decision making.
    """

    def __init__(self, detection_model, cfg: dict, device: torch.device,
                 mode: str = "rl"):
        assert mode in ("rl", "rule"), f"Unknown mode: {mode}"
        self.mode = mode
        self.detection_model = detection_model.to(device)
        self.detection_model.eval()
        self.device = device
        self.cfg = cfg
        self.class_names = cfg.get("class_names", [str(i) for i in range(15)])

        # ── Components ─────────────────────────────────────────────────
        self.env = CyberDefenseEnv(
            reward_cfg=cfg["rewards"],
            max_steps=cfg.get("max_steps", 10000),
        )

        if mode == "rl":
            self.policy = DQNAgent(cfg, device)
            rl_ckpt = cfg["rl_agent"].get("checkpoint_path")
            if rl_ckpt:
                try:
                    self.policy.load(rl_ckpt)
                    logger.info("RL policy loaded from checkpoint.")
                except FileNotFoundError:
                    logger.warning("No RL checkpoint found — using untrained policy.")
        else:
            self.policy = RuleBasedPolicy(cfg["policy"])

        self.memory   = AgentMemory(cfg["memory"])
        self.logger   = IncidentLogger(cfg["paths"]["incident_log_dir"])
        self.explainer = ThreatExplainer(self.class_names)

        # Episode statistics
        self._stats = {
            "events_processed": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "total_reward": 0.0,
            "response_times_ms": [],
        }
        logger.info(f"AgenticDefenseSystem ready | mode={mode}")

    # ── Perceive ────────────────────────────────────────────────────────
    def perceive(self, flow_window: np.ndarray) -> np.ndarray:
        """
        Run the detection model on a windowed flow sequence.

        Args:
            flow_window: (window_size, 78) normalized feature array
        Returns:
            prob_vector: (15,) softmax probabilities
        """
        x = torch.FloatTensor(flow_window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.detection_model.predict_proba(x)
        return probs.squeeze(0).cpu().numpy()

    # ── Reason & Act ────────────────────────────────────────────────────
    def act(self, prob_vector: np.ndarray,
            true_label: Optional[int] = None) -> dict:
        """
        Given a threat probability vector, select and execute an action.

        Args:
            prob_vector : (15,) softmax output from detection model
            true_label  : ground-truth label (only available in simulation)
        Returns:
            decision dict with action, confidence, reasoning, reward
        """
        t_start = time.perf_counter()

        # Build state for RL agent
        state = self.env._make_state(prob_vector)

        # Select action
        if self.mode == "rl":
            action = self.policy.select_action(state, training=False)
        else:
            action = self.policy.select_action(prob_vector)

        # Confidence and predicted class
        confidence = float(prob_vector.max())
        predicted_class = int(prob_vector.argmax())
        threat_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else "UNKNOWN"

        # Build reasoning explanation
        reasoning = self.explainer.explain(
            prob_vector=prob_vector,
            action=action,
            predicted_class=predicted_class,
            confidence=confidence,
        )

        # Compute reward if ground truth is available (simulation mode)
        reward = None
        outcome = None
        if true_label is not None:
            reward, info = self.env._compute_reward(action, true_label, prob_vector)
            outcome = info["outcome"]
            self._update_stats(outcome, reward)

        response_time_ms = (time.perf_counter() - t_start) * 1000
        self._stats["response_times_ms"].append(response_time_ms)
        self._stats["events_processed"] += 1

        decision = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "predicted_threat": threat_name,
            "predicted_class_id": predicted_class,
            "confidence": round(confidence, 4),
            "action": ACTION_NAMES[action],
            "action_id": action,
            "reasoning": reasoning,
            "reward": reward,
            "outcome": outcome,
            "response_time_ms": round(response_time_ms, 3),
            "prob_vector": prob_vector.tolist(),
        }

        # Store in memory and log
        self.memory.add(decision)
        self.logger.log(decision)

        return decision

    # ── Full Pipeline (single event) ────────────────────────────────────
    def process_event(self, flow_window: np.ndarray,
                      true_label: Optional[int] = None) -> dict:
        """End-to-end: raw flow window → decision."""
        prob_vector = self.perceive(flow_window)
        return self.act(prob_vector, true_label=true_label)

    # ── Statistics ──────────────────────────────────────────────────────
    def _update_stats(self, outcome: str, reward: float):
        self._stats["total_reward"] += reward
        if outcome == "TP":
            self._stats["tp"] += 1
        elif outcome == "FP_alert" or outcome == "FP_block":
            self._stats["fp"] += 1
        elif outcome == "FN":
            self._stats["fn"] += 1
        elif outcome == "TN":
            self._stats["tn"] += 1

    def get_summary(self) -> dict:
        s = self._stats
        tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)
        fpr       = fp / max(fp + tn, 1)
        mttr_ms   = np.mean(s["response_times_ms"]) if s["response_times_ms"] else 0.0

        return {
            "mode": self.mode,
            "events_processed": s["events_processed"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "total_reward": round(s["total_reward"], 3),
            "mean_response_time_ms": round(float(mttr_ms), 3),
        }
