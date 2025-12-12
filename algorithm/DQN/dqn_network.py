# dqn/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    # orthogonal init for conv/linear, zero bias
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class QNetwork(nn.Module):
    """
    Simple MLP Q-network for low-dimensional state inputs.
    Expects input shape: (batch, state_dim)
    """

    def __init__(self, state_size: int, action_size: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], action_size)

        # weight init
        self.apply(init_weights)

    def forward(self, x):
        # x: tensor float32, shape [B, state_size]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)