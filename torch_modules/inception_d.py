import torch
from torch import nn
from torch.nn import functional as F

from torch_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionD(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3_1 = Conv2dBNRelu(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = Conv2dBNRelu(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = Conv2dBNRelu(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = Conv2dBNRelu(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = Conv2dBNRelu(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = Conv2dBNRelu(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        return torch.cat([branch3x3, branch7x7x3, branch_pool], 1)
