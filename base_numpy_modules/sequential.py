from base_numpy_modules.module import Module


class Sequential(Module):
    def __init__(self):
        super().__init__()
        self.modules = []

    def add(self, module):
        self.modules.append(module)

    def update_output(self, input):
        for module in self.modules:
            input = module.forward(input)
        self.output = input
        return self.output

    def backward(self, input, grad_output):
        self.grad_input = grad_output
        for i in range(len(self.modules) - 1, 0, -1):
            self.grad_input = self.modules[i].backward(
                self.modules[i - 1].output, self.grad_input
            )
        self.grad_input = self.modules[0].backward(input, self.grad_input)
        return self.grad_input

    def zero_grad_parameters(self):
        for module in self.modules:
            module.zero_grad_parameters()

    def get_parameters(self):
        params = []
        for m in self.modules:
            params += m.get_parameters()
        return params

    def get_grad_parameters(self):
        grad_params = []
        for m in self.modules:
            grad_params += m.get_grad_parameters()
        return grad_params

    def __getitem__(self, item):
        return self.modules.__getitem__(item)

    def __len__(self):
        return len(self.modules)

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        self.training = False
        for module in self.modules:
            module.evaluate()
