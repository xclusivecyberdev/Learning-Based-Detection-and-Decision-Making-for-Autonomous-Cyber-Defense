"""
test_agent.py
Unit tests for the RL agent, environment, and policy components.
"""

import pytest
import numpy as np
import torch
from src.agent.rl_environment import CyberDefenseEnv, IGNORE, ALERT, BLOCK, ESCALATE
from src.agent.dqn_agent import DQNAgent, QNetwork, ReplayBuffer
from src.agent.policy import RuleBasedPolicy
from src.agent.explainer import ThreatExplainer


REWARD_CFG = {
    "true_positive_block": 1.0,
    "true_positive_alert": 0.6,
    "false_positive_block": -0.8,
    "false_positive_alert": -0.3,
    "missed_detection": -1.0,
    "correct_ignore": 0.2,
}

POLICY_CFG = {
    "confidence_thresholds": {
        "ignore": 0.3,
        "alert": 0.6,
        "block": 0.85,
        "escalate": 0.95,
    },
    "severity_map": {0: 0, 14: 4},
}


class TestCyberDefenseEnv:
    def test_reset_returns_correct_state_shape(self):
        env = CyberDefenseEnv(REWARD_CFG)
        proba = np.random.dirichlet(np.ones(15)).astype(np.float32)
        state = env.reset([(proba, 0)])
        assert state.shape == (18,), f"Expected (18,), got {state.shape}"

    def test_correct_ignore_gives_positive_reward(self):
        env = CyberDefenseEnv(REWARD_CFG)
        proba = np.zeros(15, dtype=np.float32)
        proba[0] = 1.0  # BENIGN
        state = env.reset([(proba, 0)])
        _, reward, _, info = env.step(IGNORE)
        assert reward > 0, "Correct ignore should give positive reward"
        assert info["outcome"] == "TN"

    def test_missed_detection_gives_negative_reward(self):
        env = CyberDefenseEnv(REWARD_CFG)
        proba = np.zeros(15, dtype=np.float32)
        proba[14] = 1.0  # DDoS
        state = env.reset([(proba, 14)])
        _, reward, _, info = env.step(IGNORE)
        assert reward < 0, "Missed detection should give negative reward"
        assert info["outcome"] == "FN"

    def test_episode_terminates(self):
        env = CyberDefenseEnv(REWARD_CFG, max_steps=5)
        events = [(np.random.dirichlet(np.ones(15)).astype(np.float32), i % 15)
                  for i in range(5)]
        env.reset(events)
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(ALERT)
            steps += 1
        assert steps == 5


class TestQNetwork:
    def test_output_shape(self):
        net = QNetwork(state_dim=18, action_dim=4, hidden_dim=64)
        x = torch.randn(8, 18)
        out = net(x)
        assert out.shape == (8, 4)

    def test_no_nan(self):
        net = QNetwork()
        x = torch.randn(4, 18)
        out = net(x)
        assert not torch.isnan(out).any()


class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(
                np.random.randn(18).astype(np.float32), 0, 1.0,
                np.random.randn(18).astype(np.float32), False
            )
        assert len(buf) == 50
        states, actions, rewards, nexts, dones = buf.sample(16)
        assert states.shape == (16, 18)
        assert len(actions) == 16

    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=10)
        for _ in range(20):
            buf.push(np.zeros(18), 0, 0.0, np.zeros(18), False)
        assert len(buf) == 10


class TestDQNAgent:
    def test_action_selection_shape(self, agent_cfg, device):
        agent = DQNAgent(agent_cfg, device)
        state = np.random.randn(18).astype(np.float32)
        action = agent.select_action(state, training=False)
        assert action in (0, 1, 2, 3)

    def test_greedy_action_when_eps_zero(self, agent_cfg, device):
        agent = DQNAgent(agent_cfg, device)
        agent.epsilon = 0.0
        state = np.random.randn(18).astype(np.float32)
        # Should deterministically return same action
        a1 = agent.select_action(state, training=False)
        a2 = agent.select_action(state, training=False)
        assert a1 == a2

    def test_learn_returns_zero_when_buffer_empty(self, agent_cfg, device):
        agent = DQNAgent(agent_cfg, device)
        loss = agent.learn()
        assert loss == 0.0


class TestRuleBasedPolicy:
    def test_low_confidence_gives_ignore(self):
        policy = RuleBasedPolicy(POLICY_CFG)
        proba = np.ones(15, dtype=np.float32) / 15  # uniform → confidence=0.067
        action = policy.select_action(proba)
        assert action == IGNORE

    def test_high_confidence_ddos_gives_block_or_escalate(self):
        policy = RuleBasedPolicy(POLICY_CFG)
        proba = np.zeros(15, dtype=np.float32)
        proba[14] = 0.97  # DDoS with 97% confidence
        action = policy.select_action(proba)
        assert action in (BLOCK, ESCALATE)


class TestThreatExplainer:
    def test_explains_without_error(self, sample_proba):
        class_names = [f"class_{i}" for i in range(15)]
        explainer = ThreatExplainer(class_names)
        explanation = explainer.explain(sample_proba, BLOCK, 14, 0.97)
        assert isinstance(explanation, str)
        assert len(explanation) > 20
