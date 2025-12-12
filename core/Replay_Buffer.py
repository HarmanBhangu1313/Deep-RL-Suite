"""
Replay Buffer for DQN-like algorithms.
Supports:
- Circular buffer
- (s, a, r, s', done)
- Optional N-step returns
"""

import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape, n_step: int = 1, gamma: float = 0.99):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.n_step = n_step
        self.gamma = gamma

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        # Support for N-step transitions
        self.n_step_buffer = deque(maxlen=n_step)

    def store(self, state, action, reward, next_state, done):
        """
        Stores transition using n-step logic.
        """
        # First push into temporary n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If not enough transitions yet
        if len(self.n_step_buffer) < self.n_step:
            return

        # Build N-step return transition
        R, ns, dn = self._get_n_step_info()

        # Get first transition in n-step
        s, a, _, _, _ = self.n_step_buffer[0]

        # Store to main buffer
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = R
        self.next_states[self.ptr] = ns
        self.dones[self.ptr] = dn

        # Move ptr
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_n_step_info(self):
        """Compute n-step discounted reward and last next_state."""
        R = 0.0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += r * (self.gamma ** idx)
            if d:
                return R, self.n_step_buffer[idx][3], d
        # If no done inside the window:
        return R, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

    def sample(self, batch_size: int):
        """Random minibatch sampling."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size