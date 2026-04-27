"""
explainer.py
Generates natural-language explanations for each agent decision.
Replaces LLM-based reasoning with deterministic, research-grade
explanations derived from the probability vector and action context.

This is important for PhD-level work: explainability is a core
requirement for deploying autonomous agents in high-stakes environments.
"""

import numpy as np
from src.agent.rl_environment import ACTION_NAMES, SEVERITY


THREAT_DESCRIPTIONS = {
    0: "benign traffic",
    1: "DoS Hulk flood attack",
    2: "DoS GoldenEye attack",
    3: "DoS Slowloris slow-rate attack",
    4: "DoS Slowhttptest attack",
    5: "Heartbleed vulnerability exploit",
    6: "FTP brute-force credential attack",
    7: "SSH brute-force credential attack",
    8: "Web brute-force attack",
    9: "XSS (Cross-Site Scripting) attack",
    10: "SQL Injection attack",
    11: "Network infiltration attempt",
    12: "Botnet command-and-control traffic",
    13: "Network port scan",
    14: "Distributed Denial-of-Service (DDoS) attack",
}

ACTION_RATIONALE = {
    0: "Confidence below alert threshold — no action taken to avoid false positive alert fatigue.",
    1: "Moderate confidence in threat detection — SOC analyst notified for human review.",
    2: "High-confidence threat detected — automated firewall block rule applied.",
    3: "Critical threat with high confidence — escalated to senior analyst for immediate response.",
}


class ThreatExplainer:
    def __init__(self, class_names: list):
        self.class_names = class_names

    def explain(self, prob_vector: np.ndarray, action: int,
                predicted_class: int, confidence: float) -> str:
        """
        Generate a structured natural-language explanation.
        """
        threat_name = THREAT_DESCRIPTIONS.get(predicted_class, f"class {predicted_class}")
        severity = SEVERITY.get(predicted_class, 0)

        # Top-3 alternative threats
        top3_idx = np.argsort(prob_vector)[::-1][:3]
        top3_str = ", ".join(
            f"{self.class_names[i] if i < len(self.class_names) else i} ({prob_vector[i]:.2f})"
            for i in top3_idx
        )

        action_name = ACTION_NAMES[action]
        rationale = ACTION_RATIONALE.get(action, "")

        reasoning = (
            f"Detection model classified event as {threat_name} "
            f"(confidence={confidence:.3f}, severity={severity}/4). "
            f"Top-3 candidates: [{top3_str}]. "
            f"Agent selected action={action_name}. "
            f"{rationale}"
        )
        return reasoning
