# ppo/policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ActorCritic(nn.Module):
    """
    PPO Actor–Critic network for continuous control (e.g., CarRacing-v2).
    Outputs:
      - mean: (B, action_dim)
      - std:  (B, action_dim)
      - value: (B, 1)
    """

    def __init__(self, input_channels: int, action_dim: int,
                 hidden_dim: int = 256,
                 action_std_init: float = 0.6,
                 input_shape=(96, 96)):
        """
        input_channels: number of stacked frames (e.g., 3 or 4)
        action_dim: number of continuous actions (CarRacing = 3)
        hidden_dim: size of shared FC layer
        action_std_init: initial std for Gaussian policy
        """
        super().__init__()

        self.action_dim = action_dim

        # --- CNN encoder ---
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_shape)
            conv_out = self._forward_conv(dummy)
            conv_out_size = conv_out.shape[1]

        self.fc = nn.Linear(conv_out_size, hidden_dim)

        # --- Actor ---
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(action_std_init))

        # --- Critic ---
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialization
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _forward_conv(self, x):
        """Forward pass through conv layers only."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        """
        x: (B, C, H, W), float32 [0,1]
        Returns:
          mean:  (B, action_dim)
          std:   (B, action_dim)
          value: (B, 1)
        """
        # CNN + FC
        x = self._forward_conv(x)
        x = F.relu(self.fc(x))

        # Continuous-action actor
        mean = self.actor_mean(x)                          # (B, A)
        std = torch.exp(self.log_std).expand_as(mean)      # (B, A)

        # Critic head
        value = self.critic(x)                             # (B, 1)

        return mean, std, value

    def act(self, x):
        """
        Sample an action for rollout.
        Returns sampled_action, log_prob, value
        """
        mean, std, value = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Optional: squashing (for CarRacing steering in [-1,1], throttle/brake [0,1])
        # action = torch.tanh(action)

        return action, log_prob, value