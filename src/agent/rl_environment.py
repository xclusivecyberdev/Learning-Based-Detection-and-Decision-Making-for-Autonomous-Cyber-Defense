"""
rl_environment.py
Gym-compatible cyber defense environment for training the RL agent.

State space:
    - 15-dim threat probability vector (softmax output of detection model)
    - 1-dim system load (simulated CPU/memory pressure, 0–1)
    - 1-dim hour-of-day (normalised 0–1)
    - 1-dim day-of-week (normalised 0–1)
    Total: 18-dimensional continuous state

Action space (discrete, 4 actions):
    0 = IGNORE   — no response
    1 = ALERT    — log and notify SOC analyst
    2 = BLOCK    — firewall rule injection / process isolation
    3 = ESCALATE — hand off to human analyst immediately

Reward function (research contribution):
    The reward is designed to reflect real-world SOC priorities:
    - Punish missed detections heavily (security cost >> false alarm cost)
    - Penalise false positives to avoid alert fatigue
    - Reward proportionally to threat severity

Reference:
    Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
    MIT Press. Chapter 3 (MDP formulation).
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Action indices
IGNORE   = 0
ALERT    = 1
BLOCK    = 2
ESCALATE = 3
ACTION_NAMES = {IGNORE: "IGNORE", ALERT: "ALERT", BLOCK: "BLOCK", ESCALATE: "ESCALATE"}

# Threat severity levels (used in reward shaping)
SEVERITY = {
    0: 0,   # BENIGN
    1: 2,   # DoS Hulk
    2: 2,   # DoS GoldenEye
    3: 1,   # DoS Slowloris
    4: 1,   # DoS Slowhttptest
    5: 4,   # Heartbleed
    6: 2,   # FTP-Patator
    7: 2,   # SSH-Patator
    8: 2,   # Web Attack Brute Force
    9: 2,   # Web Attack XSS
    10: 3,  # Web Attack Sql Injection
    11: 3,  # Infiltration
    12: 2,  # Bot
    13: 1,  # PortScan
    14: 4,  # DDoS
}


class CyberDefenseEnv:
    """
    Episodic cyber defense environment.

    Each episode processes a sequence of network events.
    The RL agent observes threat probabilities and chooses a response action.
    The ground-truth label determines the reward.
    """

    def __init__(self, reward_cfg: dict, max_steps: int = 1000):
        self.reward_cfg = reward_cfg
        self.max_steps = max_steps
        self.state_dim = 18
        self.action_dim = 4

        # Episode state
        self._events: Optional[list] = None   # list of (proba_vec, true_label)
        self._step_idx: int = 0
        self._episode_reward: float = 0.0
        self._episode_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    # ── Environment Interface ───────────────────────────────────────────
    def reset(self, events: list) -> np.ndarray:
        """
        Start a new episode.

        Args:
            events: list of (prob_vector: np.ndarray[15], true_label: int)
        Returns:
            initial state: np.ndarray[18]
        """
        self._events = events
        self._step_idx = 0
        self._episode_reward = 0.0
        self._episode_stats = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        return self._make_state(events[0][0])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Apply action, compute reward, advance to next event.

        Returns:
            next_state, reward, done, info
        """
        prob_vec, true_label = self._events[self._step_idx]
        reward, info = self._compute_reward(action, true_label, prob_vec)

        self._episode_reward += reward
        self._step_idx += 1
        done = self._step_idx >= min(self.max_steps, len(self._events))

        if not done:
            next_proba, _ = self._events[self._step_idx]
            next_state = self._make_state(next_proba)
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)

        info["step"] = self._step_idx
        info["episode_reward"] = self._episode_reward
        return next_state, reward, done, info

    # ── State Construction ───────────────────────────────────────────────
    def _make_state(self, prob_vec: np.ndarray) -> np.ndarray:
        """
        Concatenate threat proba (15) + system_load (1) + time features (2).
        """
        import time
        hour = (time.localtime().tm_hour) / 23.0
        dow  = (time.localtime().tm_wday) / 6.0
        # Simulate system load as a noisy signal (replace with real psutil in production)
        system_load = np.clip(np.random.normal(0.4, 0.15), 0.0, 1.0)
        state = np.concatenate([
            prob_vec.astype(np.float32),
            [np.float32(system_load)],
            [np.float32(hour)],
            [np.float32(dow)],
        ])
        return state

    # ── Reward Function ──────────────────────────────────────────────────
    def _compute_reward(self, action: int, true_label: int,
                         prob_vec: np.ndarray) -> Tuple[float, dict]:
        """
        Reward function based on action correctness and threat severity.

        Research insight: reward is shaped by severity to prevent the agent
        from learning to ignore high-severity threats even when confidence is low.
        """
        rc = self.reward_cfg
        severity = SEVERITY.get(true_label, 0)
        is_threat = true_label != 0  # 0 = BENIGN

        # --- Determine outcome ---
        if is_threat:
            if action in (ALERT, BLOCK, ESCALATE):
                # True positive — action taken on real threat
                if action == BLOCK:
                    r = rc["true_positive_block"] * (1 + severity * 0.2)
                elif action == ESCALATE:
                    r = rc.get("true_positive_escalate", 0.7) * (1 + severity * 0.1)
                else:  # ALERT
                    r = rc["true_positive_alert"] * (1 + severity * 0.1)
                self._episode_stats["tp"] += 1
                outcome = "TP"
            else:  # IGNORE
                # Missed detection — penalty scales with severity
                r = rc["missed_detection"] * (1 + severity * 0.5)
                self._episode_stats["fn"] += 1
                outcome = "FN"
        else:
            # Benign traffic
            if action == IGNORE:
                r = rc["correct_ignore"]
                self._episode_stats["tn"] += 1
                outcome = "TN"
            elif action == ALERT:
                r = rc["false_positive_alert"]
                self._episode_stats["fp"] += 1
                outcome = "FP_alert"
            else:  # BLOCK or ESCALATE on benign — worst false positive
                r = rc["false_positive_block"]
                self._episode_stats["fp"] += 1
                outcome = "FP_block"

        info = {
            "outcome": outcome,
            "true_label": true_label,
            "action": ACTION_NAMES[action],
            "reward": r,
            "severity": severity,
            "confidence": float(prob_vec.max()),
        }
        return float(r), info
