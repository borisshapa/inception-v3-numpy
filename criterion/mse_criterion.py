import numpy as np

from criterion.criterion import Criterion


class MSECriterion(Criterion):
    def __init__(self):
        super().__init__()

    def update_output(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def update_grad_input(self, input, target):
        self.grad_input = (input - target) * 2 / input.shape[0]
        return self.grad_input
