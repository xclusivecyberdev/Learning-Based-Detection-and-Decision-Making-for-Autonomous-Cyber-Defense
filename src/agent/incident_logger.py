"""
incident_logger.py
Structured JSON incident logging for every agent decision.
"""

import os
import json
import time
import logging

logger = logging.getLogger(__name__)


class IncidentLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        # Daily rotating log file
        date_str = time.strftime("%Y-%m-%d")
        self.log_path = os.path.join(log_dir, f"incidents_{date_str}.jsonl")

    def log(self, decision: dict):
        """Append decision as a JSON line to the daily log file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(decision) + "\n")
        except Exception as e:
            logger.error(f"Failed to write incident log: {e}")

    def read_today(self) -> list:
        """Read all incidents from today's log."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_stats_today(self) -> dict:
        incidents = self.read_today()
        if not incidents:
            return {"total": 0}
        from collections import Counter
        action_counts = Counter(d["action"] for d in incidents)
        threat_counts = Counter(d["predicted_threat"] for d in incidents)
        return {
            "total": len(incidents),
            "by_action": dict(action_counts),
            "top_threats": dict(threat_counts.most_common(5)),
        }
