from abc import abstractmethod


class Module:
    def __init__(self):
        self.output = None
        self.grad_input = None
        self.training = True

    def forward(self, input):
        return self.update_output(input)

    def backward(self, input, grad_output):
        self.update_grad_input(input, grad_output)
        self.acc_grad_parameters(input, grad_output)
        return self.grad_input

    @abstractmethod
    def update_output(self, input):
        pass

    @abstractmethod
    def update_grad_input(self, input, grad_output):
        pass

    @abstractmethod
    def acc_grad_parameters(self, input, grad_output):
        pass

    def zero_grad_parameters(self):
        pass

    @abstractmethod
    def get_parameters(self):
        return []

    @abstractmethod
    def get_grad_parameters(self):
        return []

    def train(self):
        self.training = True

    def evaluate(self):
        self.training = False
