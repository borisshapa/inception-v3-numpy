from base_numpy_modules.adaptive_avgpool2d import AdaptiveAvgPool2d
from base_numpy_modules.avgpool2d import AvgPool2d
from base_numpy_modules.flatten import Flatten
from base_numpy_modules.linear import Linear
from base_numpy_modules.module import Module
from base_numpy_modules.sequential import Sequential
from numpy_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionAux(Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.inception_aux = Sequential()
        self.inception_aux.add(AvgPool2d(kernel_size=5, stride=3))
        self.inception_aux.add(Conv2dBNRelu(in_channels, 128, kernel_size=1))
        self.inception_aux.add(Conv2dBNRelu(128, 768, kernel_size=5))
        self.inception_aux.add(AdaptiveAvgPool2d((1, 1)))
        self.inception_aux.add(Flatten(1))
        self.inception_aux.add(Linear(768, num_classes))

    def update_output(self, input):
        self.output = self.inception_aux.update_output(input)
        return self.output

    def backward(self, input, grad_output):
        self.grad_input = self.inception_aux.backward(input, grad_output)
        return self.grad_input

    def zero_grad_parameters(self):
        self.inception_aux.zero_grad_parameters()

    def get_parameters(self):
        return self.inception_aux.get_parameters()

    def get_grad_parameters(self):
        return self.inception_aux.get_grad_parameters()

    def train(self):
        self.training = True
        self.inception_aux.train()

    def evaluate(self):
        self.training = False
        self.inception_aux.evaluate()
