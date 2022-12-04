from typing import Any

from base_numpy_modules.batchnorm2d import BatchNorm2d
from base_numpy_modules.conv2d import Conv2d
from base_numpy_modules.module import Module
from base_numpy_modules.relu import ReLU
from base_numpy_modules.sequential import Sequential


class Conv2dBNRelu(Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any):
        super().__init__()

        self.conv_bn_relu = Sequential()
        self.conv_bn_relu.add(Conv2d(in_channels, out_channels, **kwargs))
        self.conv_bn_relu.add(BatchNorm2d(out_channels, eps=0.001))
        self.conv_bn_relu.add(ReLU())

    def update_output(self, input):
        self.output = self.conv_bn_relu.update_output(input)
        return self.output

    def backward(self, input, grad_output):
        self.grad_input = self.conv_bn_relu.backward(input, grad_output)
        return self.grad_input

    def get_parameters(self):
        return self.conv_bn_relu.get_parameters()

    def get_grad_parameters(self):
        return self.conv_bn_relu.get_grad_parameters()

    def train(self):
        self.conv_bn_relu.train()

    def evaluate(self):
        self.conv_bn_relu.evaluate()
