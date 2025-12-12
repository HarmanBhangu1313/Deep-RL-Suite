# dqn/agent.py
import random
from typing import Tuple, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.Replay_Buffer import ReplayBuffer   
from dqn.network import QNetwork               


class DQNAgent:
    """
    DQN agent with target network, soft updates, n-step support via ReplayBuffer,
    and epsilon-greedy action selection.

    Usage:
        agent = DQNAgent(state_size, action_size, config)
        agent.step(state, action, reward, next_state, done)
        action = agent.act(state, epsilon)
    """

    def __init__(self, state_size: Tuple[int, ...], action_size: int, config: Dict[str, Any]):
        # device: prefer CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.state_size = state_size
        self.action_size = action_size

        # hyperparameters (with defaults)
        self.lr = config.get("learning_rate", 1e-4)
        self.gamma = config.get("gamma", 0.99)
        self.batch_size = config.get("batch_size", 64)
        self.buffer_size = config.get("buffer_size", int(1e5))
        self.tau = config.get("tau", 1e-3)  # soft update factor
        self.update_every = config.get("update_every", 4)
        self.grad_clip = config.get("grad_clip", 10.0)

        # networks
        self.local_qnetwork = QNetwork(*state_size, action_size).to(self.device)
        self.target_qnetwork = QNetwork(*state_size, action_size).to(self.device)
        # ensure target initially matches local
        self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())

        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=self.lr)

        # replay buffer
        self.memory = ReplayBuffer(self.buffer_size, state_shape=state_size, n_step=config.get("n_step", 1), gamma=self.gamma)

        # internal step counter for periodic learning
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Save experience and (occasionally) trigger a learning step.
        state/next_state are expected as numpy arrays matching state_shape.
        """
        self.memory.store(state, action, reward, next_state, done)  # uses ReplayBuffer.store API

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection.
        state: raw numpy observation
        returns: int action
        """
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
        # if pixel obs are HxWxC, agent's network should expect channel-first; ensure network preprocesses
        state_tensor = state_tensor.to(self.device)

        self.local_qnetwork.eval()
        with torch.no_grad():
            q_values = self.local_qnetwork(state_tensor)
        self.local_qnetwork.train()

        q_values = q_values.cpu().numpy().squeeze(0)  # shape: (action_size,)
        if random.random() > epsilon:
            return int(np.argmax(q_values))
        else:
            return int(random.randrange(self.action_size))

    def learn(self, experiences: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """
        Update the local network using a batch of experiences.
        Expects experiences in ordering: (states, actions, rewards, next_states, dones)
        Each entry may be numpy arrays; convert to tensors here.
        """
        states, actions, rewards, next_states, dones = experiences

        # Convert to torch tensors and move to device
        states = torch.from_numpy(np.asarray(states)).float().to(self.device)
        actions = torch.from_numpy(np.asarray(actions)).long().unsqueeze(1).to(self.device)  # shape [B,1]
        rewards = torch.from_numpy(np.asarray(rewards)).float().unsqueeze(1).to(self.device)  # shape [B,1]
        next_states = torch.from_numpy(np.asarray(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.asarray(dones).astype(np.uint8)).float().unsqueeze(1).to(self.device)  # shape [B,1]

        # Compute target Q values (Double DQN optional - here simple DQN)
        # target_q_next = max_a Q_target(next_state, a)
        with torch.no_grad():
            q_targets_next = self.target_qnetwork(next_states).max(1)[0].unsqueeze(1)  # shape [B,1]
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))  # shape [B,1]

        # Compute expected Q from local network
        q_expected = self.local_qnetwork(states).gather(1, actions)  # shape [B,1]

        # Loss & optimize
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        # optional gradient clipping
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.local_qnetwork.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.local_qnetwork, self.target_qnetwork, self.tau)

    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float):
        """
        Soft update: target = tau*local + (1-tau)*target
        """
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str):
        torch.save({
            "local_state": self.local_qnetwork.state_dict(),
            "target_state": self.target_qnetwork.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.local_qnetwork.load_state_dict(checkpoint["local_state"])
        self.target_qnetwork.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])