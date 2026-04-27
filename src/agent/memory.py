"""
memory.py
Sliding-window agent memory for context-aware decision making.
Stores recent decisions so the agent can reflect on patterns.
"""

import json
import os
import logging
from collections import deque

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    Fixed-size sliding window of past decisions.
    Used to detect repeated threats from the same source and
    to persist decisions across sessions.
    """

    def __init__(self, memory_cfg: dict):
        self.window_size = memory_cfg["window_size"]
        self.persist_path = memory_cfg.get("persist_path")
        self.buffer = deque(maxlen=self.window_size)

        if self.persist_path and os.path.exists(self.persist_path):
            self._load()

    def add(self, decision: dict):
        self.buffer.append(decision)
        if self.persist_path:
            self._save()

    def get_recent(self, n: int = 10) -> list:
        return list(self.buffer)[-n:]

    def count_threat_type(self, threat_name: str) -> int:
        return sum(1 for d in self.buffer if d.get("predicted_threat") == threat_name)

    def _save(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(list(self.buffer), f, indent=2)

    def _load(self):
        with open(self.persist_path) as f:
            data = json.load(f)
        self.buffer = deque(data, maxlen=self.window_size)
        logger.info(f"Agent memory loaded: {len(self.buffer)} entries")
