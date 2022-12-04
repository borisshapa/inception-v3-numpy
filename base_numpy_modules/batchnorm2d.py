from base_numpy_modules.batchnorm2dbase import BatchNorm2dBase
from base_numpy_modules.channelwise_scaling import ChannelwiseScaling
from base_numpy_modules.module import Module
from base_numpy_modules.sequential import Sequential


class BatchNorm2d(Module):
    def __init__(self, n_out: int, alpha: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.bn = Sequential()
        self.bn.add(BatchNorm2dBase(alpha, eps))
        self.bn.add(ChannelwiseScaling(n_out))
        self.eps = eps

    def update_output(self, input):
        self.output = self.bn.update_output(input)
        return self.output

    def backward(self, input, grad_output):
        self.grad_input = self.bn.backward(input, grad_output)
        return self.grad_input

    def zero_grad_parameters(self):
        self.bn.zero_grad_parameters()

    def get_parameters(self):
        return self.bn.get_parameters()

    def get_grad_parameters(self):
        return self.bn.get_grad_parameters()
