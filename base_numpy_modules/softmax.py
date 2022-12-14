import numpy as np

from base_numpy_modules.module import Module


class Softmax(Module):
    def __init__(self):
        super().__init__()

    def update_output(self, input):
        exp = np.exp(input)
        exp_sum = np.sum(exp, axis=1)[:, np.newaxis]
        self.output = exp / exp_sum
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = self.output * (
            grad_output - (self.output * grad_output).sum(axis=1)[:, np.newaxis]
        )
        return self.grad_input
