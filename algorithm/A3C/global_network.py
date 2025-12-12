# a3c/global_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    # orthogonal init for conv/linear, zero bias
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class GlobalNetwork(nn.Module):
    """
    CNN + actor-critic heads for A3C.
    - Input: tensor shape [B, C, H, W] (C = stacked grayscale frames, e.g., 4)
    - Outputs:
        logits: [B, action_size]   (raw logits for categorical sampling)
        value:  [B, 1]             (state-value estimate)
    """

    def __init__(self, in_channels: int, action_size: int, last_linear_size: int = 128):
        super().__init__()
        # conv body — small kernels/strides to reduce spatial dims quickly for 84x84 inputs.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # use adaptive pooling to avoid hardcoding flatten size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # works well for 84x84 -> 7x7 after convs
        conv_out_size = 7 * 7 * 64

        self.fc = nn.Linear(conv_out_size, last_linear_size)

        # actor & critic heads
        self.actor = nn.Linear(last_linear_size, action_size)  # logits
        self.critic = nn.Linear(last_linear_size, 1)           # state value

        # initialization
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        """
        x: float tensor [B, C, H, W], values typically in [0,1] (i.e., uint8/255 normalized)
        returns: logits [B, action_size], value [B, 1]
        """
        # conv body
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # adaptive pooling + flatten
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)  # [B, conv_out_size]

        x = F.relu(self.fc(x))

        logits = self.actor(x)        # raw logits (no softmax here)
        value = self.critic(x).unsqueeze(-1)  # ensure shape [B, 1], critic already returns [B,1] usually

        return logits, value