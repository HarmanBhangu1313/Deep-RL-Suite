# ppo/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent

# Example default hyperparams — override by passing config dict
DEFAULTS = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_epochs": 10,
    "mini_batch_size": 64,
    "ppo_clip": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else "cpu"),
}

class Memory:
    def __init__(self):
        self.clear()
    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    def __init__(self, obs_space, action_space, actor_critic_class, config: dict = None):
        """
        obs_space, action_space: gym spaces
        actor_critic_class: class (ActorCritic) — will be instantiated here
        config: hyperparameters dict (overrides DEFAULTS)
        """
        cfg = DEFAULTS.copy()
        if config:
            cfg.update(config)
        self.cfg = cfg
        self.device = torch.device(cfg["device"])

        # infer action dimension (continuous vector)
        self.num_actions = int(action_space.shape[0])
        in_channels = obs_space.shape[2] if len(obs_space.shape) == 3 else 1  # HWC -> channels last
        # instantiate networks
        self.actor_critic = actor_critic_class(input_channels=in_channels, action_dim=self.num_actions).to(self.device)
        self.old_actor_critic = actor_critic_class(input_channels=in_channels, action_dim=self.num_actions).to(self.device)
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())
        self.old_actor_critic.eval()

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=cfg["lr"])

        # action bounds (tensors on device)
        self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)

        self.memory = Memory()

        # convenience aliases
        self.gamma = cfg["gamma"]
        self.lam = cfg["gae_lambda"]
        self.ppo_epochs = cfg["ppo_epochs"]
        self.mini_batch_size = cfg["mini_batch_size"]
        self.ppo_clip = cfg["ppo_clip"]
        self.value_coef = cfg["value_coef"]
        self.entropy_coef = cfg["entropy_coef"]
        self.max_grad_norm = cfg["max_grad_norm"]

    def select_action(self, obs_tensor):
        """
        obs_tensor: torch tensor, shape (C,H,W) or (1,C,H,W). Assumed already normalized float32.
        Returns: action (numpy), logprob (tensor on device), value (tensor on device)
        Note: do NOT clip action before computing logprob.
        """
        self.old_actor_critic.eval()
        x = obs_tensor
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            mean, std, value = self.old_actor_critic(x)   # mean [1,A], std [1,A], value [1,1]
            base_dist = Normal(mean, std)
            dist = Independent(base_dist, 1)
            action = dist.sample()                        # [1,A]
            logprob = dist.log_prob(action)               # [1]
        # clip for env execution (do not use clipped action for logprob math)
        action_exec = torch.clamp(action, self.action_low, self.action_high)
        return action_exec.squeeze(0).cpu().numpy(), logprob.squeeze(0).cpu(), value.squeeze(0).cpu()

    def store_transition(self, state, action, logprob, reward, done, value):
        """
        Store CPU tensors/values. Convert to CPU tensors for compact storage.
        state: torch tensor CHW (no batch dim) or numpy (converted)
        action: numpy array or torch tensor (will be stored as float32 tensor)
        logprob/value: torch tensors (will be detached and moved to cpu)
        reward: scalar float
        done: bool
        """
        # ensure state is tensor and CHW (no batch)
        if isinstance(state, np.ndarray):
            s = torch.from_numpy(state).float()
        else:
            s = state.detach().cpu().squeeze(0) if state.dim() == 4 and state.size(0) == 1 else state.detach().cpu()
        self.memory.states.append(s)
        self.memory.actions.append(torch.tensor(action, dtype=torch.float32))
        self.memory.logprobs.append(logprob.detach().cpu())
        self.memory.rewards.append(float(reward))
        self.memory.dones.append(bool(done))
        self.memory.values.append(value.detach().cpu())

    def compute_gae(self, last_value):
        """
        last_value: torch tensor scalar (on CPU) or float (value for last next_state)
        Returns tensors (advantages, returns) on CPU
        """
        rewards = self.memory.rewards
        values = [v.item() for v in self.memory.values]  # floats
        dones = self.memory.dones
        advantages = []
        gae = 0.0
        # bootstrap value
        last_v = last_value.item() if isinstance(last_value, torch.Tensor) else float(last_value)
        values_plus = values + [last_v]
        for step in reversed(range(len(rewards))):
            mask = 0.0 if dones[step] else 1.0
            delta = rewards[step] + self.gamma * values_plus[step + 1] * mask - values_plus[step]
            gae = delta + self.gamma * self.lam * mask * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values)]
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        ret_t = torch.tensor(returns, dtype=torch.float32)
        return adv_t, ret_t

    def ppo_update(self):
        mem = self.memory
        if len(mem) == 0:
            return

        # compute last_value for bootstrap
        with torch.no_grad():
            if mem.dones[-1]:
                last_val = torch.tensor(0.0)
            else:
                last_state = mem.states[-1].to(self.device)
                if last_state.dim() == 3:
                    last_state = last_state.unsqueeze(0)
                _, _, last_val = self.actor_critic(last_state.to(self.device))
                last_val = last_val.squeeze(0).cpu()

        advantages, returns = self.compute_gae(last_val)
        # normalize advantages on CPU then move to device
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # stack tensors (states/actions/logprobs)
        states = torch.stack(mem.states)                       # (N, C, H, W)
        actions = torch.stack([torch.tensor(a, dtype=torch.float32) for a in mem.actions])  # (N, A)
        old_logprobs = torch.stack(mem.logprobs).squeeze(-1)   # (N,) or (N,1) -> (N,)
        returns = returns
        advantages = advantages

        # move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        dataset_size = states.size(0)
        # if dataset smaller than mini-batch, do at least one minibatch
        mb_size = max(1, min(self.mini_batch_size, dataset_size))

        for epoch in range(self.ppo_epochs):
            # shuffle indexes
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, mb_size):
                end = start + mb_size
                mb_idx = indices[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                mean, std, value = self.actor_critic(mb_states)
                base_dist = Normal(mean, std)
                dist = Independent(base_dist, 1)
                mb_logprobs = dist.log_prob(mb_actions)
                mb_entropy = dist.entropy().mean()

                # ratio (pi / pi_old)
                ratios = torch.exp(mb_logprobs - mb_old_logprobs)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value = value.squeeze(1)
                critic_loss = F.mse_loss(value, mb_returns)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * mb_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # sync old policy
        self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())
        # clear memory
        self.memory.clear()