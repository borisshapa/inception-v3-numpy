import numpy as np

from base_numpy_modules.module import Module
from base_numpy_modules.utils import broadcast_channel_dim


class BatchNorm2dBase(Module):
    def __init__(self, alpha=0.9, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def update_output(self, input):
        if self.training:
            mean = np.mean(input, axis=(0, 2, 3))
            var = np.var(input, axis=(0, 2, 3))
            self.output = (input - broadcast_channel_dim(mean)) / np.sqrt(
                broadcast_channel_dim(var) + self.eps
            )
            self.moving_mean = (
                self.moving_mean
                if self.moving_mean is not None
                else np.zeros_like(mean)
            ) * self.alpha + mean * (1 - self.alpha)
            self.moving_variance = (
                self.moving_variance
                if self.moving_variance is not None
                else np.zeros_like(var)
            ) * self.alpha + var * (1 - self.alpha)
        else:
            self.output = (input - broadcast_channel_dim(self.moving_mean)) / np.sqrt(
                broadcast_channel_dim(self.moving_variance) + self.eps
            )
        return self.output

    def update_grad_input(self, input, grad_output):
        mean = np.mean(input, axis=(0, 2, 3), keepdims=True)
        var = np.var(input, axis=(0, 2, 3), keepdims=True)
        b = input.shape[0] * input.shape[2] * input.shape[3]

        df_dvar = -((input - mean) * grad_output).sum(axis=(0, 2, 3), keepdims=True) / (
            2 * (var + self.eps) * np.sqrt(var + self.eps)
        )

        dvar_dmean = -2 * (input - mean).sum(axis=(0, 2, 3), keepdims=True) / b
        df_dmean = (-grad_output / np.sqrt(var + self.eps)).sum(
            axis=(0, 2, 3), keepdims=True
        ) + df_dvar * dvar_dmean

        self.grad_input = (
            grad_output / np.sqrt(var + self.eps)
            + (df_dmean + df_dvar * 2 * (input - mean)) / b
        )
        return self.grad_input
