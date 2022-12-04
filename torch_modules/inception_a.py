import torch
import torch.nn.functional as F
from torch import nn

from torch_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionA(nn.Module):
    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()
        self.branch1x1 = Conv2dBNRelu(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = Conv2dBNRelu(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = Conv2dBNRelu(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = Conv2dBNRelu(in_channels, 64, kernel_size=1)
        self.branch3x3_2 = Conv2dBNRelu(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = Conv2dBNRelu(96, 96, kernel_size=3, padding=1)

        self.branch_pool = Conv2dBNRelu(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = torch.cat([branch1x1, branch5x5, branch3x3, branch_pool], 1)
        return outputs
