# sac.py

import math
import random
from collections import deque, namedtuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

# ---------- Settings / Hyperparameters ----------
SEED = 42
ENV_ID = "CarRacing-v2"
NUM_EPISODES = 200            # reduce/increase as needed
MAX_STEPS_PER_EP = 1000
REPLAY_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005                   # soft update coefficient
LR = 3e-4
HIDDEN_DIM = 256
INITIAL_RANDOM_STEPS = 4000   # collect some experience before updates
POLICY_UPDATE_FREQ = 1        # update every step (1) or fewer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 1

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------- Environment (create before dims) ----------
env = gym.make(ENV_ID)
# Gymnasium: reset() -> (obs, info), step() -> (obs, reward, terminated, truncated, info)
obs_space = env.observation_space
act_space = env.action_space

# Flatten observation if needed (e.g., images)
if hasattr(obs_space, "shape"):
    STATE_DIM = int(np.prod(obs_space.shape))
else:
    STATE_DIM = obs_space.shape[0]
ACTION_DIM = act_space.shape[0] if hasattr(act_space, "shape") else 1
ACTION_LOW = act_space.low
ACTION_HIGH = act_space.high

# ---------- Replay Buffer ----------
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([b.state for b in batch])
        actions = np.stack([b.action for b in batch])
        rewards = np.array([b.reward for b in batch], dtype=np.float32).reshape(-1, 1)
        next_states = np.stack([b.next_state for b in batch])
        dones = np.array([b.done for b in batch], dtype=np.float32).reshape(-1, 1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ---------- Networks ----------
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mean(x)
        log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        return mu, std, log_std

    def sample(self, state):
        """
        Returns:
          action: squashed action in (-1,1)
          log_prob: log probability of the squashed action (sum over action dims), shape (B,1)
          pre_tanh: pre-squash variable z (useful for debugging)
          mu: policy mean (unsquashed)
        """
        mu, std, log_std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()                 # reparam trick
        action = torch.tanh(z)
        # log_prob correction for tanh-squash
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mu

    def deterministic(self, state):
        mu, _, _ = self.forward(state)
        return torch.tanh(mu)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q = nn.Linear(hidden, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

# ---------- SAC Agent ----------
class SACAgent:
    def __init__(self, state_dim, action_dim, action_low, action_high, device=DEVICE):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        # networks
        self.policy = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        # target networks
        self.q1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # optimizers
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=LR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=LR)

        # replay
        self.replay = ReplayBuffer(REPLAY_SIZE)

        # automatic entropy tuning (learn log_alpha)
        self.target_entropy = -action_dim  # heuristic: -|A|
        self.log_alpha = torch.tensor(math.log(0.2), requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=LR)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        """
        state: raw observation, possibly multi-dim (we assume flattened input)
        evaluate: if True, return deterministic action (policy mean)
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action = self.policy.deterministic(state)
            else:
                action, _, _, _ = self.policy.sample(state)
        action = action.cpu().numpy()[0]
        # scale action from (-1,1) to env action range
        scaled = self._scale_action(action)
        return scaled

    def _scale_action(self, action):
        # action currently in (-1,1). Map to [low, high]
        low = self.action_low
        high = self.action_high
        return low + (action + 1.0) * 0.5 * (high - low)

    def _unscale_action(self, action):
        # map env action to (-1,1)
        low = self.action_low
        high = self.action_high
        return 2 * (action - low) / (high - low) - 1

    def update(self, batch_size=BATCH_SIZE, gamma=GAMMA, tau=TAU):
        if len(self.replay) < batch_size:
            return {}

        # sample batch and move to device
        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(self._unscale_action(actions)).to(self.device)  # store env actions; unscale to (-1,1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 1) compute target Q
        with torch.no_grad():
            next_actions, next_log_pi, _, _ = self.policy.sample(next_states)
            # next_actions are in (-1,1) — critics expect that same space
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_pi
            q_target = rewards + (1.0 - dones) * gamma * q_next

        # 2) update Q networks
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # 3) update policy network
        new_actions, log_pi, _, _ = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha.detach() * log_pi - q_new).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # 4) adjust alpha (temperature)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 5) soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item()
        }

# ---------- Utilities ----------
def preprocess_observation(obs):
    """
    Flatten and normalize observations to [-1,1] roughly.
    For image observations we flatten; better approach: use conv nets.
    """
    arr = np.array(obs, dtype=np.float32)
    arr = arr.reshape(-1)  # flatten
    # simple normalization: scale [0,255] images to [0,1]; clip otherwise
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr

# ---------- Training Loop ----------
def train():
    agent = SACAgent(STATE_DIM, ACTION_DIM, ACTION_LOW, ACTION_HIGH, device=DEVICE)
    total_steps = 0
    episode_rewards = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset(seed=SEED + ep)  # Gymnasium reset returns (obs, info)
        state = preprocess_observation(obs)
        ep_reward = 0.0
        done = False
        step = 0

        while not done and step < MAX_STEPS_PER_EP:
            if total_steps < INITIAL_RANDOM_STEPS:
                # random action for initial exploration (in env range)
                action = env.action_space.sample()
                scaled_action = action
                # store unscaled shape expected later (we will unscale when using)
            else:
                scaled_action = agent.select_action(state, evaluate=False)  # already scaled to env range

            next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
            next_state = preprocess_observation(next_obs)
            done_flag = bool(terminated or truncated)

            # push to replay buffer (store states and actions in env scale)
            agent.replay.push(state, scaled_action, float(reward), next_state, float(done_flag))

            # update
            if total_steps >= INITIAL_RANDOM_STEPS:
                train_info = agent.update(BATCH_SIZE)

            state = next_state
            ep_reward += reward
            total_steps += 1
            step += 1

        episode_rewards.append(ep_reward)

        # logging
        if ep % PRINT_EVERY == 0:
            avg_last = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"[Episode {ep}/{NUM_EPISODES}] reward: {ep_reward:.2f} avg_last_10: {avg_last:.2f} total_steps: {total_steps}")

    env.close()
    return agent, episode_rewards

if __name__ == "__main__":
    start = time.time()
    agent, rewards = train()
    print("Training done. Total time: {:.1f}s".format(time.time() - start))
    # optionally save models
    torch.save(agent.policy.state_dict(), "sac_policy.pth")
    torch.save(agent.q1.state_dict(), "sac_q1.pth")
    torch.save(agent.q2.state_dict(), "sac_q2.pth")