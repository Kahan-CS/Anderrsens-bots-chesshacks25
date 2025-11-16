# policy_model_resnet.py
import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class PolicyNetRes(nn.Module):
    def __init__(self, policy_size=4672):
        super().__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(17, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 8 residual blocks (AlphaZero uses 19–39)
        self.res_layers = nn.Sequential(
            *[ResidualBlock(128) for _ in range(8)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, policy_size)
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.res_layers(x)
        x = self.policy_head(x)
        return x
