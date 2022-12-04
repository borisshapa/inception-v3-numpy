from typing import Tuple

from base_numpy_modules.module import Module
from base_numpy_modules.utils import avg_pool_forward, avg_pool_backward


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        self.output_size = output_size

    def get_kernel_and_stride(self, input_size: Tuple[int, int]):
        stride_h = input_size[0] // self.output_size[0]
        stride_w = input_size[1] // self.output_size[1]
        kernel_size_h = input_size[0] - (self.output_size[0] - 1) * stride_h
        kernel_size_w = input_size[1] - (self.output_size[1] - 1) * stride_w
        return kernel_size_h, kernel_size_w, stride_h, stride_w

    def update_output(self, input):
        kernel_size_h, kernel_size_w, stride_h, stride_w = self.get_kernel_and_stride(
            (input.shape[2], input.shape[3])
        )

        self.output = avg_pool_forward(
            input,
            self.output_size,
            padding=(0, 0),
            kernel_size=(kernel_size_h, kernel_size_w),
            stride=(stride_h, stride_w),
        )
        return self.output

    def update_grad_input(self, input, grad_output):
        kernel_size_h, kernel_size_w, stride_h, stride_w = self.get_kernel_and_stride(
            (input.shape[2], input.shape[3])
        )

        self.grad_input = avg_pool_backward(
            input,
            grad_output,
            kernel_size=(kernel_size_h, kernel_size_w),
            padding=(0, 0),
            stride=(stride_h, stride_w),
        )
        return self.grad_input
