from typing import Any

from torch import nn


class Conv2dBNRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.conv_bn_relu(input)
