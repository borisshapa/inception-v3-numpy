from base_numpy_modules.flatten import Flatten
from base_numpy_modules.module import Module
from base_numpy_modules.sequential import Sequential
from base_numpy_modules.maxpool2d import MaxPool2d
from base_numpy_modules.adaptive_avgpool2d import AdaptiveAvgPool2d
from base_numpy_modules.dropout import Dropout
from base_numpy_modules.linear import Linear
from numpy_modules.conv2d_bn_relu import Conv2dBNRelu
from numpy_modules.inception_a import InceptionA
from numpy_modules.inception_b import InceptionB
from numpy_modules.inception_c import InceptionC
from numpy_modules.inception_d import InceptionD
from numpy_modules.inception_e import InceptionE
from numpy_modules.inception_aux import InceptionAux


class InceptionV3(Module):
    def __init__(self, num_classes: int, dropout: float = 0.5, use_aux: bool = False, in_channels: int = 3):
        super().__init__()
        self.use_aux = use_aux

        self.before_aux = Sequential()
        self.before_aux.add(Conv2dBNRelu(in_channels, 32, kernel_size=3, stride=2))
        self.before_aux.add(Conv2dBNRelu(32, 32, kernel_size=3))
        self.before_aux.add(Conv2dBNRelu(32, 64, kernel_size=3, stride=1, padding=1))
        self.before_aux.add(MaxPool2d(kernel_size=3, stride=2))
        self.before_aux.add(Conv2dBNRelu(64, 80, kernel_size=1))
        self.before_aux.add(Conv2dBNRelu(80, 192, kernel_size=3))
        self.before_aux.add(MaxPool2d(kernel_size=3, stride=2))
        self.before_aux.add(InceptionA(192, 32))
        self.before_aux.add(InceptionA(256, 64))
        self.before_aux.add(InceptionA(288, 64))
        self.before_aux.add(InceptionB(288))
        self.before_aux.add(InceptionC(768, 128))
        self.before_aux.add(InceptionC(768, 160))
        self.before_aux.add(InceptionC(768, 160))
        self.before_aux.add(InceptionC(768, 192))

        self.aux_logits = InceptionAux(768, num_classes)

        self.after_aux = Sequential()
        self.after_aux.add(InceptionD(768))
        self.after_aux.add(InceptionE(1280))
        self.after_aux.add(InceptionE(2048))
        self.after_aux.add(AdaptiveAvgPool2d((1, 1)))
        self.after_aux.add(Dropout(p=dropout))
        self.after_aux.add(Flatten(start_dim=1))
        self.after_aux.add(Linear(2048, num_classes))

    def update_output(self, x):
        x = self.before_aux.update_output(x)

        aux = (
            self.aux_logits.update_output(x) if self.use_aux and self.training else None
        )

        x = self.after_aux.update_output(x)
        self.output = (x, aux)
        return self.output

    def backward(self, input, grad_output):
        if self.use_aux:
            grad_output, grad_aux = grad_output
            grad_aux = self.aux_logits.backward(self.before_aux.output, grad_aux)
        grad_output = self.after_aux.backward(self.before_aux.output, grad_output)
        if self.use_aux:
            grad_output += grad_aux
        self.grad_input = self.before_aux.backward(input, grad_output)
        return self.grad_input

    def zero_grad_parameters(self):
        self.before_aux.zero_grad_parameters()
        self.aux_logits.zero_grad_parameters()
        self.after_aux.zero_grad_parameters()

    def get_parameters(self):
        return (
            self.before_aux.get_parameters()
            + self.aux_logits.get_parameters()
            + self.after_aux.get_parameters()
        )

    def get_grad_parameters(self):
        return (
            self.before_aux.get_grad_parameters()
            + self.aux_logits.get_grad_parameters()
            + self.after_aux.get_grad_parameters()
        )

    def train(self):
        self.training = True
        self.before_aux.train()
        self.aux_logits.train()
        self.aux_logits.train()

    def evaluate(self):
        self.training = False
        self.before_aux.evaluate()
        self.aux_logits.evaluate()
        self.after_aux.evaluate()
