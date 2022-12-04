import torch
from torch import nn
from torch.nn import functional as F
from torch_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionC(nn.Module):
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        self.branch1x1 = Conv2dBNRelu(in_channels, 192, kernel_size=1)

        self.branch7x7_1 = Conv2dBNRelu(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7_2 = Conv2dBNRelu(
            channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3)
        )
        self.branch7x7_3 = Conv2dBNRelu(
            channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0)
        )

        self._branch7x7_1 = Conv2dBNRelu(in_channels, channels_7x7, kernel_size=1)
        self._branch7x7_2 = Conv2dBNRelu(
            channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0)
        )
        self._branch7x7_3 = Conv2dBNRelu(
            channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3)
        )
        self._branch7x7_4 = Conv2dBNRelu(
            channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0)
        )
        self._branch7x7_5 = Conv2dBNRelu(
            channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3)
        )

        self.branch_pool = Conv2dBNRelu(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        _branch7x7 = self._branch7x7_1(x)
        _branch7x7 = self._branch7x7_2(_branch7x7)
        _branch7x7 = self._branch7x7_3(_branch7x7)
        _branch7x7 = self._branch7x7_4(_branch7x7)
        _branch7x7 = self._branch7x7_5(_branch7x7)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        return torch.cat([branch1x1, branch7x7, _branch7x7, branch_pool], 1)
