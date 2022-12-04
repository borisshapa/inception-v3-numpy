import torch
from torch import nn
from torch.nn import functional as F
from torch_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = Conv2dBNRelu(in_channels, 128, kernel_size=1)
        self.conv2 = Conv2dBNRelu(128, 768, kernel_size=5)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv1(x)
        x = self.conv2(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
