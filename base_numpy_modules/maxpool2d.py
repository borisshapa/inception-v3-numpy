import numpy as np

from base_numpy_modules.module import Module


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, padding: int = 0, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.grad_input = None
        self.stride = kernel_size if stride is None else stride

    def update_output(self, input):
        input = np.pad(
            input, ((0,), (0,), (self.padding,), (self.padding,)), mode="minimum"
        )

        bs, channels, input_h, input_w = input.shape

        h = int((input_h - self.kernel_size) / self.stride) + 1
        w = int((input_w - self.kernel_size) / self.stride) + 1

        self.output = np.zeros((bs, channels, h, w))
        for sample in range(bs):
            for c in range(channels):
                for i in range(h):
                    for j in range(w):
                        ii = i * self.stride
                        jj = j * self.stride
                        slice = input[
                            sample,
                            c,
                            ii : ii + self.kernel_size,
                            jj : jj + self.kernel_size,
                        ]
                        self.output[sample, c, i, j] = np.max(slice)
        return self.output

    def update_grad_input(self, input, grad_output):
        bs, channels, h, w = grad_output.shape
        _, _, input_h, input_w = input.shape
        self.grad_input = np.zeros(
            (
                bs,
                channels,
                input_h + 2 * self.padding,
                input_w + 2 * self.padding,
            )
        )
        zero_pad = np.pad(
            input,
            (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="minimum",
        )

        for sample in range(bs):
            for c in range(channels):
                for i in range(h):
                    for j in range(w):
                        ii = i * self.stride
                        jj = j * self.stride
                        slice = zero_pad[
                            sample,
                            c,
                            ii : ii + self.kernel_size,
                            jj : jj + self.kernel_size,
                        ]
                        self.grad_input[
                            sample,
                            c,
                            ii : ii + self.kernel_size,
                            jj : jj + self.kernel_size,
                        ] += grad_output[sample, c, i, j] * ((slice == np.max(slice)) if slice.shape[0] != 0 and slice.shape[1] != 0 else 0)
        self.grad_input = self.grad_input[
            :,
            :,
            self.padding : self.padding + input_h,
            self.padding : self.padding + input_w,
        ]
        return self.grad_input
