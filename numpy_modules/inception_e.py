import numpy as np

from base_numpy_modules.avgpool2d import AvgPool2d
from base_numpy_modules.module import Module
from numpy_modules.conv2d_bn_relu import Conv2dBNRelu


class InceptionE(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.branch1x1 = Conv2dBNRelu(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = Conv2dBNRelu(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = Conv2dBNRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = Conv2dBNRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self._branch3x3_1 = Conv2dBNRelu(in_channels, 448, kernel_size=1)
        self._branch3x3_2 = Conv2dBNRelu(448, 384, kernel_size=3, padding=1)
        self._branch3x3_3a = Conv2dBNRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self._branch3x3_3b = Conv2dBNRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.avg_pool = AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool = Conv2dBNRelu(in_channels, 192, kernel_size=1)

    def update_output(self, input):
        branch1x1 = self.branch1x1.update_output(input)

        branch3x3 = self.branch3x3_1.update_output(input)
        branch3x3 = [
            self.branch3x3_2a.update_output(branch3x3),
            self.branch3x3_2b.update_output(branch3x3),
        ]
        branch3x3 = np.concatenate(branch3x3, 1)

        _branch3x3 = self._branch3x3_1.update_output(input)
        _branch3x3 = self._branch3x3_2.update_output(_branch3x3)
        _branch3x3 = [
            self._branch3x3_3a.update_output(_branch3x3),
            self._branch3x3_3b.update_output(_branch3x3),
        ]
        _branch3x3 = np.concatenate(_branch3x3, 1)

        branch_pool = self.avg_pool.update_output(input)
        branch_pool = self.branch_pool.update_output(branch_pool)

        self.output = np.concatenate([branch1x1, branch3x3, _branch3x3, branch_pool], 1)
        return self.output

    def backward(self, input, grad_output):
        grad_branch1x1, grad_branch3x3, grad__branch3x3, grad_branch_pool = np.split(
            grad_output, axis=1, indices_or_sections=[320, 1088, 1856]
        )

        grad_branch1x1 = self.branch1x1.backward(input, grad_branch1x1)
        grad_branch3x3_a, grad_branch3x3_b = np.split(
            grad_branch3x3, axis=1, indices_or_sections=[384]
        )

        b3x3_1_out = self.branch3x3_1.output
        grad_branch3x3 = self.branch3x3_2a.backward(
            b3x3_1_out, grad_branch3x3_a
        ) + self.branch3x3_2b.backward(b3x3_1_out, grad_branch3x3_b)
        grad_branch3x3 = self.branch3x3_1.backward(input, grad_branch3x3)

        grad__branch3x3_a, grad__branch3x3_b = np.split(
            grad__branch3x3, axis=1, indices_or_sections=[384]
        )
        _b3x3_2_out = self._branch3x3_2.output
        grad__branch3x3 = self._branch3x3_3a.backward(
            _b3x3_2_out, grad__branch3x3_a
        ) + self._branch3x3_3b.backward(_b3x3_2_out, grad__branch3x3_b)
        grad__branch3x3 = self._branch3x3_2.backward(
            self._branch3x3_1.output, grad__branch3x3
        )
        grad__branch3x3 = self._branch3x3_1.backward(input, grad__branch3x3)

        grad_branch_pool = self.branch_pool.backward(
            self.avg_pool.output, grad_branch_pool
        )
        grad_branch_pool = self.avg_pool.backward(input, grad_branch_pool)
        self.grad_input = grad_branch_pool + grad__branch3x3 + grad_branch3x3 + grad_branch1x1
        return self.grad_input

    def zero_grad_parameters(self):
        self.branch1x1.zero_grad_parameters()
        self.branch3x3_1.zero_grad_parameters()
        self.branch3x3_2a.zero_grad_parameters()
        self.branch3x3_2b.zero_grad_parameters()
        self._branch3x3_1.zero_grad_parameters()
        self._branch3x3_2.zero_grad_parameters()
        self._branch3x3_3a.zero_grad_parameters()
        self._branch3x3_3b.zero_grad_parameters()
        self.avg_pool.zero_grad_parameters()
        self.branch_pool.zero_grad_parameters()

    def get_parameters(self):
        return (
            self.branch1x1.get_parameters()
            + self.branch3x3_1.get_parameters()
            + self.branch3x3_2a.get_parameters()
            + self.branch3x3_2b.get_parameters()
            + self._branch3x3_1.get_parameters()
            + self._branch3x3_2.get_parameters()
            + self._branch3x3_3a.get_parameters()
            + self._branch3x3_3b.get_parameters()
            + self.avg_pool.get_parameters()
            + self.branch_pool.get_parameters()
        )

    def get_grad_parameters(self):
        return (
            self.branch1x1.get_grad_parameters()
            + self.branch3x3_1.get_grad_parameters()
            + self.branch3x3_2a.get_grad_parameters()
            + self.branch3x3_2b.get_grad_parameters()
            + self._branch3x3_1.get_grad_parameters()
            + self._branch3x3_2.get_grad_parameters()
            + self._branch3x3_3a.get_grad_parameters()
            + self._branch3x3_3b.get_grad_parameters()
            + self.avg_pool.get_grad_parameters()
            + self.branch_pool.get_grad_parameters()
        )

    def train(self):
        self.training = True
        self.branch1x1.train()
        self.branch3x3_1.train()
        self.branch3x3_2a.train()
        self.branch3x3_2b.train()
        self._branch3x3_1.train()
        self._branch3x3_2.train()
        self._branch3x3_3a.train()
        self._branch3x3_3b.train()
        self.avg_pool.train()
        self.branch_pool.train()

    def evaluate(self):
        self.training = False
        self.branch1x1.evaluate()
        self.branch3x3_1.evaluate()
        self.branch3x3_2a.evaluate()
        self.branch3x3_2b.evaluate()
        self._branch3x3_1.evaluate()
        self._branch3x3_2.evaluate()
        self._branch3x3_3a.evaluate()
        self._branch3x3_3b.evaluate()
        self.avg_pool.evaluate()
        self.branch_pool.evaluate()
