from torch import nn
from torch.nn import Sequential

from torch_modules.conv2d_bn_relu import Conv2dBNRelu
from torch_modules.inception_a import InceptionA
from torch_modules.inception_aux import InceptionAux
from torch_modules.inception_b import InceptionB
from torch_modules.inception_c import InceptionC
from torch_modules.inception_d import InceptionD
from torch_modules.inception_e import InceptionE


class InceptionV3(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5, use_aux: bool = False):
        super().__init__()
        self.use_aux = use_aux
        self.before_aux = Sequential(
            Conv2dBNRelu(3, 32, kernel_size=3, stride=2),
            Conv2dBNRelu(32, 32, kernel_size=3),
            Conv2dBNRelu(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Conv2dBNRelu(64, 80, kernel_size=1),
            Conv2dBNRelu(80, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            InceptionA(192, 32),
            InceptionA(256, 64),
            InceptionA(288, 64),
            InceptionB(288),
            InceptionC(768, 128),
            InceptionC(768, 160),
            InceptionC(768, 160),
            InceptionC(768, 192),
        )

        self.aux_logits = InceptionAux(768, num_classes)

        self.after_aux = Sequential(
            InceptionD(768),
            InceptionE(1280),
            InceptionE(2048),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(p=dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.before_aux(x)
        aux = self.aux_logits(x) if self.training and self.use_aux else None
        x = self.after_aux(x)
        return x, aux
