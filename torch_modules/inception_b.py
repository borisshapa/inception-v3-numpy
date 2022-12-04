import torch
from torch import nn
from torch.nn import functional as F
from torch_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = Conv2dBNRelu(in_channels, 384, kernel_size=3, stride=2)
        self._branch3x3_1 = Conv2dBNRelu(in_channels, 64, kernel_size=1)
        self._branch3x3_2 = Conv2dBNRelu(64, 96, kernel_size=3, padding=1)
        self._branch3x3_3 = Conv2dBNRelu(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        _branch3x3 = self._branch3x3_1(x)
        _branch3x3 = self._branch3x3_2(_branch3x3)
        _branch3x3 = self._branch3x3_3(_branch3x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        return torch.cat([branch3x3, _branch3x3, branch_pool], 1)
