import numpy as np

from base_numpy_modules.log_softmax import LogSoftmax
from criterion.criterion import Criterion


class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.eps = 1e-15

    def update_output(self, input, target):
        input = self.log_softmax.update_output(input)
        batch_size = input.shape[0]
        self.output = -(target * input).sum() / batch_size
        return self.output

    def update_grad_input(self, input, target):
        batch_size = input.shape[0]
        self.grad_input = -target / batch_size
        self.grad_input = self.log_softmax.backward(input, self.grad_input)
        return self.grad_input
