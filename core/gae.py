"""
Generalized Advantage Estimation (GAE-Lambda)
Used for PPO and A3C.

Given:
    rewards[t]
    values[t]
    dones[t]
    gamma
    lam
Returns:
    advantages[t]
    returns[t]
"""

import numpy as np


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values: [T+1]  (bootstrap with last value)
    dones:  [T]

    returns advantage[T], returns[T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns