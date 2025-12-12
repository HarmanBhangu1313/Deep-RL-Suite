"""
General utilities used across algorithms.
"""

import numpy as np
import torch
import random
import os


def set_seed(seed: int):
    """Make experiments reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(obs, device):
    """
    Convert observation to PyTorch tensor.
    Handles pixel observations: (H, W, C) -> (C, H, W)
    """
    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim == 3:
        # channel last -> channel first
        obs = np.transpose(obs, (2, 0, 1))

    return torch.from_numpy(obs).float().unsqueeze(0).to(device)


def discount_cumsum(x, gamma):
    """
    Compute discounted cumulative sums:
    y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ...
    """
    y = np.zeros_like(x)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + gamma * running
        y[t] = running
    return y


def init_weights(m):
    """Orthogonal initialization for linear/conv layers (good for PPO/A3C)."""
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)