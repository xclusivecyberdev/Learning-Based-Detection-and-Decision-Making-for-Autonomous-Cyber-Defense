"""
rl_trainer.py
Training loop for the DQN agent using the CyberDefenseEnv.

This runs AFTER the detection model is trained. It uses the trained
detection model to generate threat probability vectors from the test
set, then trains the RL agent to learn optimal response policies.

Experimental design:
    - Train RL agent on 70% of test-set events
    - Evaluate on remaining 30% (held-out simulation set)
    - Compare: DQN agent vs. Rule-based baseline vs. Random policy
    - Metrics: F1, FPR, MTTR, cumulative reward
"""

import os
import json
import logging
import numpy as np
import torch
from tqdm import tqdm

from src.agent.dqn_agent import DQNAgent
from src.agent.rl_environment import CyberDefenseEnv

logger = logging.getLogger(__name__)


def collect_events(detection_model, data_loader, device) -> list:
    """
    Run detection model over a DataLoader to produce (prob_vector, true_label)
    pairs used as the RL environment's event stream.
    """
    detection_model.eval()
    events = []
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            probs = detection_model.predict_proba(X)
            probs_np = probs.cpu().numpy()
            y_np = y.numpy()
            for p, label in zip(probs_np, y_np):
                events.append((p, int(label)))
    logger.info(f"Collected {len(events)} events for RL training")
    return events


def train_rl_agent(detection_model, train_loader, val_loader,
                   cfg: dict, device: torch.device) -> DQNAgent:
    """
    Full RL training loop.

    Args:
        detection_model : trained BiLSTM+CNN model
        train_loader    : DataLoader for RL training episodes
        val_loader      : DataLoader for RL evaluation
        cfg             : full config dict
        device          : torch device
    Returns:
        trained DQNAgent
    """
    agent = DQNAgent(cfg, device)
    env   = CyberDefenseEnv(cfg["rewards"])

    n_episodes = cfg["rl_agent"].get("train_episodes", 50)
    eval_every = cfg["rl_agent"].get("eval_every", 10)
    rl_ckpt_path = cfg["paths"]["rl_checkpoint"]

    # Collect events from training set
    train_events = collect_events(detection_model, train_loader, device)
    val_events   = collect_events(detection_model, val_loader, device)

    episode_rewards = []
    best_val_reward = float("-inf")

    for episode in range(1, n_episodes + 1):
        # Shuffle events each episode to prevent overfitting to order
        np.random.shuffle(train_events)
        state = env.reset(train_events)

        total_reward = 0.0
        done = False
        step = 0

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            total_reward += reward
            step += 1

        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)

        logger.info(
            f"Episode {episode:03d}/{n_episodes} | "
            f"reward={total_reward:.2f} | avg10={avg_reward:.2f} | "
            f"eps={agent.epsilon:.3f} | steps={step}"
        )
        print(
            f"RL Episode {episode:03d}/{n_episodes}  "
            f"reward={total_reward:.2f}  avg10={avg_reward:.2f}  "
            f"eps={agent.epsilon:.3f}"
        )

        # Periodic evaluation on validation events
        if episode % eval_every == 0:
            val_reward = _evaluate_agent(agent, env, val_events)
            print(f"  [Eval] val_reward={val_reward:.2f}")
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                agent.save(rl_ckpt_path)
                print(f"  ✓ Best RL agent saved (val_reward={val_reward:.2f})")

    # Save training curve
    history = {"episode_rewards": episode_rewards}
    history_path = os.path.join(os.path.dirname(rl_ckpt_path), "rl_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return agent


def _evaluate_agent(agent, env, events: list) -> float:
    """Run agent on events without exploration, return total reward."""
    state = env.reset(events)
    total_reward = 0.0
    done = False
    while not done:
        action = agent.select_action(state, training=False)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward
