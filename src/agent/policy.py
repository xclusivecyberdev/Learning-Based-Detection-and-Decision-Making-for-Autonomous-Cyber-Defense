"""
policy.py
Rule-based baseline policy for comparison against the RL agent.

This implements the threshold-based decision policy used as the
BASELINE in ablation studies. Compare its performance metrics
against the DQN agent to quantify the benefit of learned policies.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Action constants (mirrors rl_environment.py)
IGNORE   = 0
ALERT    = 1
BLOCK    = 2
ESCALATE = 3


class RuleBasedPolicy:
    """
    Threshold-based policy (baseline).

    Decision logic:
        confidence < ignore_thresh          → IGNORE
        ignore_thresh ≤ confidence < alert  → ALERT
        alert ≤ confidence < block          → BLOCK
        confidence ≥ block                  → ESCALATE if severity ≥ 3, else BLOCK

    This mirrors common SOC playbook threshold rules.
    """

    def __init__(self, policy_cfg: dict):
        thresholds = policy_cfg["confidence_thresholds"]
        self.ignore_thresh   = thresholds["ignore"]
        self.alert_thresh    = thresholds["alert"]
        self.block_thresh    = thresholds["block"]
        self.escalate_thresh = thresholds["escalate"]
        self.severity_map    = policy_cfg.get("severity_map", {})

    def select_action(self, prob_vector: np.ndarray) -> int:
        """
        Args:
            prob_vector: (15,) softmax probabilities from detection model
        Returns:
            action int: 0=IGNORE, 1=ALERT, 2=BLOCK, 3=ESCALATE
        """
        confidence     = float(prob_vector.max())
        predicted_class = int(prob_vector.argmax())
        # Get class name from index (requires mapping — use index as string key)
        severity = self.severity_map.get(predicted_class, 1)

        if confidence < self.ignore_thresh:
            return IGNORE
        elif confidence < self.alert_thresh:
            return ALERT
        elif confidence < self.block_thresh:
            return BLOCK
        else:
            # High confidence — escalate if high severity threat
            if severity >= 3:
                return ESCALATE
            return BLOCK
