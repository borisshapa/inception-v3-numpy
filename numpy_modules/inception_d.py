import numpy as np

from base_numpy_modules.maxpool2d import MaxPool2d
from base_numpy_modules.module import Module
from base_numpy_modules.sequential import Sequential
from numpy_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionD(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch3x3 = Sequential()
        self.branch3x3.add(Conv2dBNRelu(in_channels, 192, kernel_size=1))
        self.branch3x3.add(Conv2dBNRelu(192, 320, kernel_size=3, stride=2))

        self.branch7x7x3 = Sequential()
        self.branch7x7x3.add(Conv2dBNRelu(in_channels, 192, kernel_size=1))
        self.branch7x7x3.add(Conv2dBNRelu(192, 192, kernel_size=(1, 7), padding=(0, 3)))
        self.branch7x7x3.add(Conv2dBNRelu(192, 192, kernel_size=(7, 1), padding=(3, 0)))
        self.branch7x7x3.add(Conv2dBNRelu(192, 192, kernel_size=3, stride=2))

        self.max_pool = MaxPool2d(kernel_size=3, stride=2)

    def update_output(self, input):
        branch3x3 = self.branch3x3.update_output(input)
        branch7x7x3 = self.branch7x7x3.update_output(input)
        branch_pool = self.max_pool.update_output(input)
        return np.concatenate([branch3x3, branch7x7x3, branch_pool], axis=1)

    def backward(self, input, grad_output):
        grad_branch3x3, grad_branch7x7x3, grad_branch_pool = np.split(
            grad_output, axis=1, indices_or_sections=[320, 512]
        )

        grad_branch_pool = self.max_pool.backward(input, grad_branch_pool)
        grad_branch7x7x3 = self.branch7x7x3.backward(input, grad_branch7x7x3)
        grad_branch3x3 = self.branch3x3.backward(input, grad_branch3x3)
        return grad_branch3x3 + grad_branch7x7x3 + grad_branch_pool

    def zero_grad_parameters(self):
        self.branch3x3.zero_grad_parameters()
        self.branch7x7x3.zero_grad_parameters()

    def get_parameters(self):
        return self.branch3x3.get_parameters() + self.branch7x7x3.get_parameters()

    def get_grad_parameters(self):
        return (
            self.branch3x3.get_grad_parameters()
            + self.branch7x7x3.get_grad_parameters()
        )
