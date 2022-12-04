from typing import Tuple

import numpy as np
import numpy.typing as npt


def get_kernel_and_stride(input_size: Tuple[int, int], output_size: Tuple[int, int]):
    stride_h = input_size[0] // output_size[0]
    stride_w = input_size[1] // output_size[1]

    kernel_size_h = input_size[0] - (output_size[0] - 1) * stride_h
    kernel_size_w = input_size[1] - (output_size[1] - 1) * stride_w

    return kernel_size_h, kernel_size_w, stride_h, stride_w


def avg_pool_forward(
    input: npt.NDArray,
    output_size: Tuple[int, int],
    padding: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> npt.NDArray:
    bs, channels, input_h, input_w = input.shape
    output_h, output_w = output_size
    padding_h, padding_w = padding
    kernel_size_h, kernel_size_w = kernel_size
    stride_h, stride_w = stride

    input = np.pad(input, ((0,), (0,), (padding_h,), (padding_w,)))
    output = np.zeros((bs, channels, output_h, output_w))
    for sample in range(bs):
        for c in range(channels):
            for i in range(output_h):
                for j in range(output_w):
                    slice = input[
                        sample,
                        c,
                        i * stride_h : i * stride_h + kernel_size_h,
                        j * stride_w : j * stride_w + kernel_size_w,
                    ]
                    output[sample, c, i, j] = np.mean(slice)
    return output


def avg_pool_backward(
    input: npt.NDArray,
    grad_output: npt.NDArray,
    padding: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
):
    bs, channels, h, w = grad_output.shape
    _, _, input_h, input_w = input.shape
    padding_h, padding_w = padding
    stride_h, stride_w = stride
    kernel_size_h, kernel_size_w = kernel_size

    grad_input = np.zeros(
        (
            bs,
            channels,
            input_h + 2 * padding_h,
            input_w + 2 * padding_w,
        )
    )
    for sample in range(bs):
        for c in range(channels):
            for i in range(h):
                for j in range(w):
                    grad_input[
                        sample,
                        c,
                        i * stride_h : i * stride_h + kernel_size_h,
                        j * stride_w : j * stride_w + kernel_size_w,
                    ] += (
                        grad_output[sample, c, i, j]
                        * np.ones((kernel_size_h, kernel_size_w))
                        / (kernel_size_h * kernel_size_w)
                    )
    grad_input = grad_input[
        :,
        :,
        padding_h : padding_h + input_h,
        padding_w : padding_w + input_w,
    ]
    return grad_input

def broadcast_channel_dim(a: npt.NDArray) -> npt.NDArray:
    return a[np.newaxis, :, np.newaxis, np.newaxis]
