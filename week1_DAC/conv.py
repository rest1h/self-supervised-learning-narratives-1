import torch
from torch import nn
from typing import Tuple


def conv_block(in_channels: int, out_channels: int, kernel_size: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def pooling_block(kernel_size: Tuple, out_channels: int, pooling: nn.MaxPool2d):
    return nn.Sequential(
        pooling(kernel_size=kernel_size),
        nn.BatchNorm2d(out_channels),
    )


def conv_net_block(in_channels: int, out_channels: int, kernel_size: int, pooling_size: Tuple):
    return nn.Sequential(
        conv_block(in_channels, out_channels, kernel_size),
        conv_block(out_channels, out_channels, kernel_size),
        conv_block(out_channels, out_channels, kernel_size),
        pooling_block(pooling_size, out_channels, nn.MaxPool2d),
    )


def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU()
    )


class MNISTNetwork(nn.Module):
    def __init__(self):
        super(MNISTNetwork, self).__init__()
        self.net1 = conv_net_block(1, 64, 3, (2, 2))
        self.net2 = conv_net_block(64, 128, 3, (2, 2))
        self.net3 = nn.Sequential(
            conv_block(128, 10, 1), pooling_block((2, 2), 10, nn.AvgPool2d)
        )
        self.net4 = nn.Sequential(mlp(10, 10), mlp(10, 10))
        self.softmax = nn.Softmax(dim=1)
        self._eta = nn.Parameter(torch.zeros(1, dtype=torch.float))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in: torch.Tensor) -> torch.tensor:

        x = self.net1(x_in)
        x = self.net2(x)
        x = self.net3(x)
        x = x.view(-1, 10)
        x = self.net4(x)
        return self.softmax(x)

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        self._eta = eta

    def _fix_eta(self):
        eta = torch.abs(self._eta)
        self._eta = (eta * 0.99) if eta > 1.0 else eta
