import numpy as np

from base_numpy_modules.module import Module


class Linear(Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        stdv = 1 / np.sqrt(n_in)
        self.w = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

    def update_output(self, input):
        self.output = np.dot(input, self.w.T) + self.b
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = np.dot(grad_output, self.w)
        return self.grad_input

    def acc_grad_parameters(self, input, grad_output):
        self.grad_w = np.dot(grad_output.T, input)
        self.grad_b = np.dot(np.ones(input.shape[0]), grad_output)

    def zero_grad_parameters(self):
        self.grad_w.fill(0)
        self.grad_b.fill(0)

    def get_parameters(self):
        return [self.w, self.b]

    def get_grad_parameters(self):
        return [self.grad_w, self.grad_b]
