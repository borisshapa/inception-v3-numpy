import numpy as np

from base_numpy_modules.avgpool2d import AvgPool2d
from base_numpy_modules.module import Module
from base_numpy_modules.sequential import Sequential
from numpy_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionC(Module):
    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()
        self.branch1x1 = Conv2dBNRelu(in_channels, 192, kernel_size=1)

        self.branch7x7 = Sequential()
        self.branch7x7.add(Conv2dBNRelu(in_channels, channels_7x7, kernel_size=1))
        self.branch7x7.add(
            Conv2dBNRelu(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        )
        self.branch7x7.add(
            Conv2dBNRelu(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        self._branch7x7 = Sequential()
        self._branch7x7.add(Conv2dBNRelu(in_channels, channels_7x7, kernel_size=1))
        self._branch7x7.add(
            Conv2dBNRelu(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        )
        self._branch7x7.add(
            Conv2dBNRelu(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        )
        self._branch7x7.add(
            Conv2dBNRelu(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        )
        self._branch7x7.add(
            Conv2dBNRelu(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = Sequential()
        self.branch_pool.add(AvgPool2d(kernel_size=3, stride=1, padding=1))
        self.branch_pool.add(Conv2dBNRelu(in_channels, 192, kernel_size=1))

    def update_output(self, input):
        branch1x1 = self.branch1x1.update_output(input)
        branch7x7 = self.branch7x7.update_output(input)
        _branch7x7 = self._branch7x7.update_output(input)
        branch_pool = self.branch_pool.update_output(input)
        self.output = np.concatenate([branch1x1, branch7x7, _branch7x7, branch_pool], 1)
        return self.output

    def backward(self, input, grad_output):
        grad_branch1x1, grad_branch7x7, grad__branch7x7, grad_branch_pool = np.split(
            grad_output, axis=1, indices_or_sections=[192, 384, 576]
        )

        grad_branch_pool = self.branch_pool.backward(input, grad_branch_pool)
        grad__branch7x7 = self._branch7x7.backward(input, grad__branch7x7)
        grad_branch7x7 = self.branch7x7.backward(input, grad_branch7x7)
        grad_branch1x1 = self.branch1x1.backward(input, grad_branch1x1)

        self.grad_input = grad_branch1x1 + grad_branch7x7 + grad__branch7x7 + grad_branch_pool
        return self.grad_input

    def zero_grad_parameters(self):
        self.branch1x1.zero_grad_parameters()
        self.branch7x7.zero_grad_parameters()
        self._branch7x7.zero_grad_parameters()
        self.branch_pool.zero_grad_parameters()

    def get_parameters(self):
        return (
            self.branch1x1.get_parameters()
            + self.branch7x7.get_parameters()
            + self._branch7x7.get_parameters()
            + self.branch_pool.get_parameters()
        )

    def get_grad_parameters(self):
        return (
            self.branch1x1.get_grad_parameters()
            + self.branch7x7.get_grad_parameters()
            + self._branch7x7.get_grad_parameters()
            + self.branch_pool.get_grad_parameters()
        )

    def train(self):
        self.training = True
        self.branch1x1.train()
        self.branch7x7.train()
        self._branch7x7.train()
        self.branch_pool.train()

    def evaluate(self):
        self.training = False
        self.branch1x1.evaluate()
        self.branch7x7.evaluate()
        self._branch7x7.evaluate()
        self.branch_pool.evaluate()
