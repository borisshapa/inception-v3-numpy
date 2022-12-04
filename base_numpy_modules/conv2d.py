from typing import Union, Tuple

import numpy as np
import scipy

from base_numpy_modules.module import Module


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        stride: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size_h, self.kernel_size_w = kernel_size, kernel_size
        else:
            self.kernel_size_h, self.kernel_size_w = kernel_size

        if isinstance(padding, int):
            self.padding_h, self.padding_w = padding, padding
        else:
            self.padding_h, self.padding_w = padding

        self.stride = stride

        stdv = 1.0 / np.sqrt(in_channels)
        self.w = np.random.uniform(
            -stdv,
            stdv,
            size=(out_channels, in_channels, self.kernel_size_h, self.kernel_size_w),
        )
        self.b = np.random.uniform(-stdv, stdv, size=(out_channels,))
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

    def update_output(self, input):
        bs, c, input_h, input_w = input.shape
        zero_pad = np.pad(
            input,
            (
                (0, 0),
                (0, 0),
                (self.padding_h, self.padding_h),
                (self.padding_w, self.padding_w),
            ),
        )

        h = 1 + (input_h + 2 * self.padding_h - self.kernel_size_h) // self.stride
        w = 1 + (input_w + 2 * self.padding_w - self.kernel_size_w) // self.stride

        self.output = np.empty((bs, self.out_channels, h, w))
        for batch_ind in range(bs):
            sample = zero_pad[batch_ind]
            for c in range(self.out_channels):
                for i in range(h):
                    for j in range(w):
                        ii = i * self.stride
                        jj = j * self.stride
                        slice = sample[
                            :,
                            ii : ii + self.kernel_size_h,
                            jj : jj + self.kernel_size_w,
                        ]
                        self.output[batch_ind, c, i, j] = (
                            np.multiply(slice, self.w[c]).sum() + self.b[c]
                        )
        return self.output

    def update_grad_input(self, input, grad_output):
        _, _, input_h, input_w = input.shape
        bs, _, h, w = grad_output.shape

        self.grad_input = np.zeros_like(input)
        for batch_ind in range(bs):
            grad_sample = np.zeros(
                (
                    self.in_channels,
                    input_h + 2 * self.padding_h,
                    input_w + 2 * self.padding_w,
                )
            )
            for c in range(self.out_channels):
                for i in range(h):
                    for j in range(w):
                        ii = i * self.stride
                        jj = j * self.stride
                        grad_sample[
                            :,
                            ii : ii + self.kernel_size_h,
                            jj : jj + self.kernel_size_w,
                        ] += (
                            self.w[c] * grad_output[batch_ind, c, i, j]
                        )

            self.grad_input[batch_ind, :, :, :] = grad_sample[
                :,
                self.padding_h : self.padding_h + input_h,
                self.padding_w : self.padding_w + input_w,
            ]
        return self.grad_input

    def acc_grad_parameters(self, input, grad_output):
        _, _, input_h, input_w = input.shape
        bs, _, h, w = grad_output.shape

        zero_pad = np.pad(
            input,
            (
                (0, 0),
                (0, 0),
                (self.padding_h, self.padding_h),
                (self.padding_w, self.padding_w),
            ),
        )
        for batch_ind in range(bs):
            sample = zero_pad[batch_ind]
            for c in range(self.out_channels):
                for i in range(h):
                    for j in range(w):
                        ii = i * self.stride
                        jj = j * self.stride
                        slice = sample[
                            :,
                            ii : ii + self.kernel_size_h,
                            jj : jj + self.kernel_size_w,
                        ]
                        self.grad_w[c, :, :, :] += (
                            slice * grad_output[batch_ind, c, i, j]
                        )
                        self.grad_b[c] += grad_output[batch_ind, c, i, j]
        return self.grad_input

    def zero_grad_parameters(self):
        self.grad_w.fill(0)
        self.grad_b.fill(0)

    def get_parameters(self):
        return [self.w, self.b]

    def get_grad_parameters(self):
        return [self.grad_w, self.grad_b]
