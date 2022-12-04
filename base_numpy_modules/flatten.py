from base_numpy_modules.module import Module


class Flatten(Module):
    def __init__(self, start_dim: int = 1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim

    def update_output(self, input):
        new_shape = []
        rest = 1
        for ind, dim in enumerate(input.shape):
            if ind < self.start_dim:
                new_shape.append(dim)
            else:
                rest *= dim
        if len(new_shape) < len(input.shape):
            new_shape.append(rest)
        self.output = input.reshape(new_shape)
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = grad_output.reshape(input.shape)
        return self.grad_input
