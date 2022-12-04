import numpy as np

from base_numpy_modules.module import Module
from base_numpy_modules.utils import broadcast_channel_dim


class ChannelwiseScaling(Module):
    def __init__(self, n_out):
        super().__init__()

        stdv = 1.0 / np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.grad_gamma = np.zeros_like(self.gamma)
        self.grad_beta = np.zeros_like(self.beta)

    def update_output(self, input):
        self.output = input * broadcast_channel_dim(self.gamma) + broadcast_channel_dim(
            self.beta
        )
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = grad_output * broadcast_channel_dim(self.gamma)
        return self.grad_input

    def acc_grad_parameters(self, input, grad_output):
        self.grad_beta = np.sum(grad_output, axis=(0, 2, 3))
        self.grad_gamma = np.sum(grad_output * input, axis=(0, 2, 3))

    def zero_grad_parameters(self):
        self.grad_gamma.fill(0)
        self.grad_beta.fill(0)

    def get_parameters(self):
        return [self.gamma, self.beta]

    def get_grad_parameters(self):
        return [self.grad_gamma, self.grad_beta]
