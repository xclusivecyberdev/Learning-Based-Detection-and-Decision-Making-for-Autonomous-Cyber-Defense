"""
conftest.py — Pytest fixtures shared across all tests.
"""

import pytest
import torch
import numpy as np
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def cfg():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs/train_config.yaml"
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def agent_cfg():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs/agent_config.yaml"
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_batch():
    """(batch=4, window=50, features=78) random tensor."""
    return torch.randn(4, 50, 78)


@pytest.fixture
def sample_labels():
    """4 random class labels in [0, 14]."""
    return torch.randint(0, 15, (4,))


@pytest.fixture
def sample_proba():
    """Random 15-class probability vector (sums to 1)."""
    p = np.random.dirichlet(np.ones(15))
    return p.astype(np.float32)


@pytest.fixture
def sample_state(sample_proba):
    """18-dim RL state vector."""
    return np.concatenate([sample_proba, [0.4, 0.5, 0.3]]).astype(np.float32)
