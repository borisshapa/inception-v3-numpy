import numpy as np

from base_numpy_modules.avgpool2d import AvgPool2d
from base_numpy_modules.module import Module
from base_numpy_modules.sequential import Sequential
from numpy_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionA(Module):
    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()
        self.branch1x1 = Conv2dBNRelu(in_channels, 64, kernel_size=1)

        self.branch5x5 = Sequential()
        self.branch5x5.add(Conv2dBNRelu(in_channels, 48, kernel_size=1))
        self.branch5x5.add(Conv2dBNRelu(48, 64, kernel_size=5, padding=2))

        self.branch3x3 = Sequential()
        self.branch3x3.add(Conv2dBNRelu(in_channels, 64, kernel_size=1))
        self.branch3x3.add(Conv2dBNRelu(64, 96, kernel_size=3, padding=1))
        self.branch3x3.add(Conv2dBNRelu(96, 96, kernel_size=3, padding=1))

        self.branch_pool = Sequential()
        self.branch_pool.add(AvgPool2d(kernel_size=3, padding=1, stride=1))
        self.branch_pool.add(Conv2dBNRelu(in_channels, pool_features, kernel_size=1))

    def update_output(self, input):
        branch1x1 = self.branch1x1.update_output(input)
        branch5x5 = self.branch5x5.update_output(input)
        branch3x3 = self.branch3x3.update_output(input)
        branch_pool = self.branch_pool.update_output(input)

        self.output = np.concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=1)
        return self.output

    def backward(self, input, grad_output):
        grad_branch1x1, grad_branch5x5, grad_branch3x3, grad_branch_pool = np.split(
            grad_output, axis=1, indices_or_sections=[64, 128, 224]
        )

        grad_branch_pool = self.branch_pool.backward(input, grad_branch_pool)
        grad_branch3x3 = self.branch3x3.backward(input, grad_branch3x3)
        grad_branch5x5 = self.branch5x5.backward(input, grad_branch5x5)
        grad_branch1x1 = self.branch1x1.backward(input, grad_branch1x1)
        self.grad_input = (
            grad_branch_pool + grad_branch3x3 + grad_branch5x5 + grad_branch1x1
        )
        return self.grad_input

    def zero_grad_parameters(self):
        self.branch1x1.zero_grad_parameters()
        self.branch3x3.zero_grad_parameters()
        self.branch5x5.zero_grad_parameters()
        self.branch_pool.zero_grad_parameters()

    def get_parameters(self):
        return (
            self.branch1x1.get_parameters()
            + self.branch3x3.get_parameters()
            + self.branch5x5.get_parameters()
            + self.branch_pool.get_parameters()
        )

    def get_grad_parameters(self):
        return (
            self.branch1x1.get_grad_parameters()
            + self.branch3x3.get_grad_parameters()
            + self.branch5x5.get_grad_parameters()
            + self.branch_pool.get_grad_parameters()
        )

    def train(self):
        self.training = True
        self.branch1x1.train()
        self.branch3x3.train()
        self.branch5x5.train()
        self.branch_pool.train()

    def evaluate(self):
        self.training = False
        self.branch1x1.evaluate()
        self.branch3x3.evaluate()
        self.branch5x5.evaluate()
        self.branch_pool.evaluate()