import numpy as np

from base_numpy_modules.module import Module


class LogSoftmax(Module):
    def __init__(self):
        super().__init__()

    def update_output(self, input):
        sum_exp = np.sum(np.exp(input), axis=1)
        log_sum_exp = np.log(sum_exp)[:, np.newaxis]
        self.output = input - log_sum_exp
        return self.output

    def update_grad_input(self, input, grad_output):
        exp = np.exp(input)
        exp_sum = np.sum(exp, axis=1)[:, np.newaxis]
        softmax = exp / exp_sum
        self.grad_input = grad_output - softmax * grad_output.sum(axis=1)[:, np.newaxis]
        return self.grad_input
