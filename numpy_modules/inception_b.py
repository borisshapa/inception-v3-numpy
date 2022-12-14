import numpy as np

from base_numpy_modules.maxpool2d import MaxPool2d
from base_numpy_modules.module import Module
from base_numpy_modules.sequential import Sequential
from numpy_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionB(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = Conv2dBNRelu(in_channels, 384, kernel_size=3, stride=2)
        self._branch3x3 = Sequential()
        self._branch3x3.add(Conv2dBNRelu(in_channels, 64, kernel_size=1))
        self._branch3x3.add(Conv2dBNRelu(64, 96, kernel_size=3, padding=1))
        self._branch3x3.add(Conv2dBNRelu(96, 96, kernel_size=3, stride=2))
        self.max_pool = MaxPool2d(kernel_size=3, stride=2)

    def update_output(self, input):
        branch3x3 = self.branch3x3.update_output(input)
        _branch3x3 = self._branch3x3.update_output(input)
        branch_pool = self.max_pool.update_output(input)
        self.output = np.concatenate([branch3x3, _branch3x3, branch_pool], axis=1)
        return self.output

    def backward(self, input, grad_output):
        grad_branch3x3, grad__branch3x3, grad_branch_pool = np.split(
            grad_output, axis=1, indices_or_sections=[384, 480]
        )

        grad_branch_pool = self.max_pool.backward(input, grad_branch_pool)
        grad__branch3x3 = self._branch3x3.backward(input, grad__branch3x3)
        grad_branch3x3 = self.branch3x3.backward(input, grad_branch3x3)

        self.grad_input = grad_branch3x3 + grad__branch3x3 + grad_branch_pool
        return self.grad_input

    def zero_grad_parameters(self):
        self.branch3x3.zero_grad_parameters()
        self._branch3x3.zero_grad_parameters()

    def get_parameters(self):
        return self.branch3x3.get_parameters() + self._branch3x3.get_parameters()

    def get_grad_parameters(self):
        return (
            self.branch3x3.get_grad_parameters() + self._branch3x3.get_grad_parameters()
        )

    def train(self):
        self.training = True
        self.branch3x3.train()
        self._branch3x3.train()

    def evaluate(self):
        self.training = False
        self.branch3x3.evaluate()
        self._branch3x3.evaluate()
