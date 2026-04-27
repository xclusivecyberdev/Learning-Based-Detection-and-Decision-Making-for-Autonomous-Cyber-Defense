"""
dqn_agent.py
Deep Q-Network (DQN) agent with Experience Replay and Target Network.

This is the core research contribution: replacing rule-based response
decisions with a learned policy that adapts over time.

Research question:
    Does a DQN-based response policy reduce Mean Time to Respond (MTTR)
    and False Positive Rate (FPR) compared to static threshold-based rules?

Architecture:
    - Online network: predicts Q(s, a) for action selection
    - Target network: provides stable TD targets (updated every N steps)
    - Replay buffer: breaks temporal correlation between training samples

Reference:
    Mnih, V., et al. (2015). Human-level control through deep reinforcement
    learning. Nature, 518, 529–533. https://doi.org/10.1038/nature14236
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
import logging
from collections import deque
from typing import Tuple

logger = logging.getLogger(__name__)


# ── Q-Network ──────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    Dueling DQN architecture:
        - Shared feature extractor
        - Value stream V(s)
        - Advantage stream A(s, a)
        - Q(s, a) = V(s) + A(s, a) - mean(A(s, :))

    Dueling helps the agent distinguish states where action choice matters
    (active threat) from those where it doesn't (benign traffic).

    Reference:
        Wang, Z., et al. (2016). Dueling network architectures for deep
        reinforcement learning. ICML 2016.
    """

    def __init__(self, state_dim: int = 18, action_dim: int = 4,
                 hidden_dim: int = 128):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(x)
        value     = self.value_stream(features)        # (batch, 1)
        advantage = self.advantage_stream(features)    # (batch, action_dim)
        # Combine: Q = V + (A - mean(A))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ── Replay Buffer ──────────────────────────────────────────────────────────
class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ──────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Dueling DQN agent with:
        - ε-greedy exploration with exponential decay
        - Target network with periodic hard update
        - Experience replay
        - Gradient clipping

    The agent's state is the combined output of:
        (detection model softmax probabilities, system context features)
    """

    def __init__(self, cfg: dict, device: torch.device = None):
        self.cfg = cfg["rl_agent"]
        self.reward_cfg = cfg["rewards"]
        self.device = device or torch.device("cpu")

        state_dim  = self.cfg["state_dim"]
        action_dim = self.cfg["action_dim"]
        hidden_dim = self.cfg["hidden_dim"]

        # Networks
        self.online_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=self.cfg["lr"]
        )
        self.replay_buffer = ReplayBuffer(self.cfg["replay_buffer_size"])

        self.gamma   = self.cfg["gamma"]
        self.epsilon = self.cfg["epsilon_start"]
        self.eps_end = self.cfg["epsilon_end"]
        self.eps_decay = self.cfg["epsilon_decay"]
        self.batch_size = self.cfg["batch_size"]
        self.target_update_freq = self.cfg["target_update_freq"]
        self.min_replay_size = self.cfg["min_replay_size"]

        self.steps_done = 0
        self.training_losses: list = []

        logger.info(f"DQN Agent initialised | device={self.device} | "
                    f"state_dim={state_dim} | action_dim={action_dim}")

    # ── Action Selection ────────────────────────────────────────────────
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.cfg["action_dim"])

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Learning Step ────────────────────────────────────────────────────
    def learn(self) -> float:
        """Sample a batch from replay, compute TD loss, update online net."""
        if len(self.replay_buffer) < self.min_replay_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a) for the taken actions
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: select best action with online net, evaluate with target net
        with torch.no_grad():
            best_actions = self.online_net(next_states_t).argmax(dim=1)
            next_q = self.target_net(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.SmoothL1Loss()(current_q, target_q)  # Huber loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1

        # Decay epsilon
        self.epsilon = max(
            self.eps_end, self.epsilon * self.eps_decay
        )

        # Periodic target network update
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            logger.debug(f"Target network updated at step {self.steps_done}")

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    # ── Checkpoint I/O ──────────────────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "training_losses": self.training_losses,
        }, path)
        logger.info(f"DQN Agent saved to {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint at {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.steps_done = ckpt["steps_done"]
        self.training_losses = ckpt.get("training_losses", [])
        logger.info(f"DQN Agent loaded from {path} (steps={self.steps_done})")
