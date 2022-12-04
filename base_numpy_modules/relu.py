import numpy as np

from base_numpy_modules.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def update_output(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = np.multiply(grad_output, input > 0)
        return self.grad_input
