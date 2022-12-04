class Criterion:
    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, input, target):
        return self.update_output(input, target)

    def backward(self, input, target):
        return self.update_grad_input(input, target)

    def update_output(self, input, target):
        return self.output

    def update_grad_input(self, input, target):
        return self.grad_input
