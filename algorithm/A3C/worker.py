# a3c/worker.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.nn as nn

from a3c.global_network import GlobalNetwork  # your network module
from core.utils import set_seed

class A3CWorker:
    """
    A single A3C worker (actor-critic). This can be used as:
    - a single-process actor-critic trainer (call `update_from_batch`)
    - or as the basis for a multiprocessing A3C worker (where local network params
      are synced from a shared global network, gradients are applied to the global optimizer).

    API:
      worker = A3CWorker(in_channels, action_size, config)
      action = worker.act(state)                 # state: numpy array (C,H,W) or batch
      worker.update_from_batch(batch)            # batch: dict with states, actions, rewards, next_states, dones
    """

    def __init__(self, in_channels: int, action_size: int, config: dict = None):
        config = config or {}
        # device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.lr = config.get("learning_rate", 1e-4)
        self.grad_clip = config.get("grad_clip", 0.5)

        # network - this is the worker-local network by default
        self.network = GlobalNetwork(in_channels, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def act(self, state):
        """
        Sample an action for a single state or batch of states.
        state: numpy array shape (C,H,W) or (B,C,H,W)
        returns: numpy array of actions (shape: (,) or (B,))
        """
        self.network.eval()
        # ensure batch dimension
        if isinstance(state, np.ndarray):
            if state.ndim == 3:
                state = np.expand_dims(state, 0)  # [1,C,H,W]
            state_tensor = torch.from_numpy(state).float().to(self.device) / 255.0
        else:
            # if already torch tensor
            state_tensor = state.float().to(self.device)

        with torch.no_grad():
            logits, _ = self.network(state_tensor)   # logits: [B, action_size]
            dist = Categorical(logits=logits)
            actions = dist.sample()                 # tensor shape [B]
        self.network.train()
        return actions.cpu().numpy()

    def update_from_batch(self, batch: dict):
        """
        Update network parameters from a batch of transitions.
        Expects batch as dict with keys:
          'states'       : np.array or torch tensor [B, C, H, W] (or [B, state_dim] for low-dim)
          'actions'      : np.array or tensor [B]
          'rewards'      : np.array or tensor [B]
          'next_states'  : np.array or tensor [B, ...]
          'dones'        : np.array or tensor [B] (0/1)
        Returns dict of losses for logging.
        """
        # Convert everything to torch tensors on device with correct dtypes/shapes
        states = torch.from_numpy(batch['states']).float().to(self.device)
        actions = torch.from_numpy(batch['actions']).long().to(self.device)
        rewards = torch.from_numpy(batch['rewards']).float().to(self.device)
        next_states = torch.from_numpy(batch['next_states']).float().to(self.device)
        dones = torch.from_numpy(batch['dones'].astype(np.float32)).float().to(self.device)

        # Normalize pixel inputs if necessary (if wrappers didn't)
        if states.ndim == 4 and states.max() > 2.0:  # simple heuristic for uint8 frames
            states = states / 255.0
            next_states = next_states / 255.0

        # Forward pass for current states and next states
        logits, state_values = self.network(states)            # logits: [B, A], state_values: [B,1]
        with torch.no_grad():
            _, next_state_values = self.network(next_states)   # bootstrap values [B,1]

        # Ensure shapes: [B,1] for values
        state_values = state_values.view(-1, 1)
        next_state_values = next_state_values.view(-1, 1)

        # Compute target / advantage
        # r + gamma * V(next) * (1 - done)
        targets = rewards.unsqueeze(1) + self.gamma * next_state_values * (1.0 - dones.unsqueeze(1))
        advantages = targets - state_values  # shape [B,1]

        # Policy loss (actor)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)           # shape [B]
        entropy = dist.entropy()                     # shape [B]

        # align shapes: advantages.squeeze()
        actor_loss = - (log_probs * advantages.detach().squeeze(1)).mean()
        entropy_loss = - self.entropy_coef * entropy.mean()

        # Critic loss (value)
        critic_loss = F.mse_loss(state_values, targets.detach())

        total_loss = actor_loss + critic_loss + entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
        self.optimizer.step()

        # Return scalar losses for logging
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.mean().item(),
            "total_loss": total_loss.item(),
            "avg_value": state_values.mean().item(),
        }

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path, map_location=None):
        st = torch.load(path, map_location=map_location or self.device)
        self.network.load_state_dict(st)