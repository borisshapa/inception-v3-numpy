import torch
from torch import nn
from torch.nn import functional as F

from torch_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionE(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch1x1 = Conv2dBNRelu(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = Conv2dBNRelu(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = Conv2dBNRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = Conv2dBNRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self._branch3x3_1 = Conv2dBNRelu(in_channels, 448, kernel_size=1)
        self._branch3x3_2 = Conv2dBNRelu(448, 384, kernel_size=3, padding=1)
        self._branch3x3_3a = Conv2dBNRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self._branch3x3_3b = Conv2dBNRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = Conv2dBNRelu(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        _branch3x3 = self._branch3x3_1(x)
        _branch3x3 = self._branch3x3_2(_branch3x3)
        _branch3x3 = [self._branch3x3_3a(_branch3x3), self._branch3x3_3b(_branch3x3)]
        _branch3x3 = torch.cat(_branch3x3, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch3x3, _branch3x3, branch_pool], 1)
