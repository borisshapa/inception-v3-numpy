import numpy as np

from base_numpy_modules.module import Module


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def update_output(self, input):
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.p, input.shape)
            self.output = input * self.mask / (1 - self.p)
        else:
            self.output = input
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = grad_output * self.mask / (1 - self.p)
        return self.grad_input
