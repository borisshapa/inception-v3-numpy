import numpy as np

from base_numpy_modules.module import Module
from base_numpy_modules.utils import avg_pool_forward, avg_pool_backward


class AvgPool2d(Module):
    def __init__(self, kernel_size: int, padding: int = 0, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.grad_input = None
        self.stride = kernel_size if stride is None else stride

    def get_output_dim(self, input_dim: int):
        return 1 + (input_dim + 2 * self.padding - self.kernel_size) // self.stride

    def update_output(self, input):
        output_h = self.get_output_dim(input.shape[2])
        output_w = self.get_output_dim(input.shape[3])
        self.output = avg_pool_forward(
            input,
            (output_h, output_w),
            padding=(self.padding, self.padding),
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
        )
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = avg_pool_backward(
            input,
            grad_output,
            padding=(self.padding, self.padding),
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
        )
        return self.grad_input
