import numpy as np

from base_numpy_modules.softmax import Softmax
from criterion.criterion import Criterion


class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
        self.eps = 1e-15

    def update_output(self, input, target):
        input = self.softmax.update_output(input)
        input_clamp = np.clip(input, self.eps, 1 - self.eps)
        batch_size = input_clamp.shape[0]
        self.output = -(target * np.log(input_clamp)).sum() / batch_size
        return self.output

    def update_grad_input(self, input, target):
        input_clamp = np.clip(input, self.eps, 1 - self.eps)
        batch_size = input_clamp.shape[0]
        self.grad_input = -target * (1 / input_clamp) / batch_size
        self.grad_input *= np.ma.masked_inside(input, self.eps, 1 - self.eps).mask
        self.grad_input = self.softmax.backward(input, self.grad_input)
        return self.grad_input
