import unittest

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from base_numpy_modules.adaptive_avgpool2d import AdaptiveAvgPool2d
from base_numpy_modules.avgpool2d import AvgPool2d
from base_numpy_modules.batchnorm2d import BatchNorm2d
from base_numpy_modules.batchnorm2dbase import BatchNorm2dBase
from base_numpy_modules.conv2d import Conv2d
from base_numpy_modules.flatten import Flatten
from base_numpy_modules.maxpool2d import MaxPool2d
from numpy_modules.inception_a import InceptionA
from numpy_modules.inception_aux import InceptionAux
from numpy_modules.inception_b import InceptionB
from numpy_modules.inception_c import InceptionC
from numpy_modules.inception_d import InceptionD
from numpy_modules.inception_e import InceptionE
from torch_modules.inception_a import InceptionA as TorchInceptionA
from torch_modules.inception_b import InceptionB as TorchInceptionB
from torch_modules.inception_c import InceptionC as TorchInceptionC
from torch_modules.inception_d import InceptionD as TorchInceptionD
from torch_modules.inception_e import InceptionE as TorchInceptionE
from torch_modules.inception_aux import InceptionAux as TorchInceptionAux


class Rule:
    def __init__(
        self,
        numpy_module: str,
        sequential: bool = False,
        range=None,
        add_id: bool = True,
        suffix: str = "",
    ):
        self.numpy_module = numpy_module
        self.sequential = sequential
        self.suffix = suffix
        self.range = range
        self.add_id = add_id
        if sequential and range is None:
            raise RuntimeError()


class TestLayers(unittest.TestCase):
    def set_weight_and_bias(self, custom_layer, torch_layer):
        custom_layer.w = torch_layer.weight.data.numpy()
        custom_layer.b = torch_layer.bias.data.numpy()

    def set_bn_params(self, custom_layer, torch_layer):
        custom_layer.bn[0].moving_mean = torch_layer.running_mean.numpy().copy()
        custom_layer.bn[0].moving_variance = torch_layer.running_var.numpy().copy()
        custom_layer.bn[1].gamma = torch_layer.weight.data.numpy()
        custom_layer.bn[1].beta = torch_layer.bias.data.numpy()

    def prepare_conv_bn(self, custom_layer, torch_layer):
        self.set_weight_and_bias(
            custom_layer.conv_bn_relu[0], torch_layer.conv_bn_relu[0]
        )
        self.set_bn_params(custom_layer.conv_bn_relu[1], torch_layer.conv_bn_relu[1])

    def prepare_basic_conv(self, custom_model, torch_model, rule):
        if rule.sequential:
            sequential = getattr(custom_model, rule.numpy_module)
            for i in rule.range:
                custom_layer = sequential[i]
                torch_layer = getattr(
                    torch_model,
                    f"{rule.numpy_module}_{i + 1}"
                    if rule.add_id
                    else rule.numpy_module,
                )
                self.prepare_conv_bn(custom_layer, torch_layer)
        else:
            custom_layer = getattr(custom_model, rule.numpy_module)
            torch_layer = getattr(torch_model, rule.numpy_module)
            self.prepare_conv_bn(custom_layer, torch_layer)

    def test_BatchNorm2dBase(self):
        np.random.seed(21)
        torch.manual_seed(21)

        batch_size, c, h, w = 32, 3, 4, 5
        for _ in range(100):
            alpha = 0.9

            custom_layer = BatchNorm2dBase(alpha)
            custom_layer.train()

            torch_layer = nn.BatchNorm2d(
                c, eps=custom_layer.eps, momentum=1.0 - alpha, affine=False
            )
            custom_layer.moving_mean = torch_layer.running_mean.numpy().copy()
            custom_layer.moving_variance = torch_layer.running_var.numpy().copy()

            layer_input = np.random.uniform(-5, 5, (batch_size, c, h, w)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-5, 5, (batch_size, c, h, w)).astype(
                np.float32
            )

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.update_grad_input(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad

            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-5
                )
            )
            self.assertTrue(
                np.allclose(custom_layer.moving_mean, torch_layer.running_mean.numpy())
            )

            custom_layer.moving_variance = torch_layer.running_var.numpy().copy()
            custom_layer.evaluate()
            custom_layer_output = custom_layer.update_output(layer_input)
            torch_layer.eval()
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

    def test_BatchNorm2d(self):
        np.random.seed(21)
        torch.manual_seed(21)

        batch_size, c, h, w = 32, 3, 4, 5
        for _ in range(100):
            alpha = 0.9

            custom_layer = BatchNorm2d(c, alpha)
            torch_layer = nn.BatchNorm2d(c, eps=custom_layer.eps, momentum=1.0 - alpha)

            self.set_bn_params(custom_layer, torch_layer)
            custom_layer.train()

            layer_input = np.random.uniform(-5, 5, (batch_size, c, h, w)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(-5, 5, (batch_size, c, h, w)).astype(
                np.float32
            )

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad

            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-5
                )
            )

            weight_grad, bias_grad = custom_layer.get_grad_parameters()[1]
            torch_weight_grad = torch_layer.weight.grad.data.numpy()
            torch_bias_grad = torch_layer.bias.grad.data.numpy()

            self.assertTrue(np.allclose(torch_weight_grad, weight_grad, atol=1e-5))
            self.assertTrue(np.allclose(torch_bias_grad, bias_grad, atol=1e-5))

    def test_AvgPool2d(self):
        np.random.seed(21)
        torch.manual_seed(21)

        batch_size, n_in = 2, 3
        h, w = 4, 6
        kernel_size = 3
        stride = 1
        padding = 1
        for _ in range(100):
            torch_layer = torch.nn.AvgPool2d(
                kernel_size=kernel_size, padding=padding, stride=stride
            )
            custom_layer = AvgPool2d(
                kernel_size=kernel_size, padding=padding, stride=stride
            )

            layer_input = np.random.uniform(-10, 10, (batch_size, n_in, h, w)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(
                -10,
                10,
                (
                    batch_size,
                    n_in,
                    int(1 + (h + 2 * padding - kernel_size) / stride),
                    int(1 + (w + 2 * padding - kernel_size) / stride),
                ),
            ).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.update_grad_input(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_AdaptiveAvgPool2d(self):
        np.random.seed(21)
        torch.manual_seed(21)

        batch_size, n_in = 2, 3
        h, w = 4, 6
        for _ in range(100):
            torch_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
            custom_layer = AdaptiveAvgPool2d((1, 1))

            layer_input = np.random.uniform(-10, 10, (batch_size, n_in, h, w)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(
                -10,
                10,
                (batch_size, n_in, 1, 1),
            ).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.update_grad_input(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_MaxPool2d(self):
        np.random.seed(21)
        torch.manual_seed(21)

        batch_size, n_in = 2, 3
        h, w = 4, 6
        kernel_size = 3
        stride = 1
        padding = 1
        for _ in range(100):
            torch_layer = torch.nn.MaxPool2d(
                kernel_size=kernel_size, padding=padding, stride=stride
            )
            custom_layer = MaxPool2d(
                kernel_size=kernel_size, padding=padding, stride=stride
            )

            layer_input = np.random.uniform(-10, 10, (batch_size, n_in, h, w)).astype(
                np.float32
            )
            next_layer_grad = np.random.uniform(
                -10,
                10,
                (
                    batch_size,
                    n_in,
                    int(1 + (h + 2 * padding - kernel_size) / stride),
                    int(1 + (w + 2 * padding - kernel_size) / stride),
                ),
            ).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.update_grad_input(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_Conv2d_stride1(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 5, 6
        kernel_size = 3
        for _ in range(100):
            torch_layer = nn.Conv2d(n_in, n_out, kernel_size, padding=1)
            custom_layer = Conv2d(n_in, n_out, kernel_size, padding=1)
            self.set_weight_and_bias(custom_layer, torch_layer)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(-1, 1, (bs, n_out, h, w)).astype(
                np.float32
            )

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.update_grad_input(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

            custom_layer.acc_grad_parameters(layer_input, next_layer_grad)
            weight_grad = custom_layer.grad_w
            bias_grad = custom_layer.grad_b
            torch_weight_grad = torch_layer.weight.grad.data.numpy()
            torch_bias_grad = torch_layer.bias.grad.data.numpy()

            self.assertTrue(np.allclose(torch_weight_grad, weight_grad, atol=1e-6))
            self.assertTrue(np.allclose(torch_bias_grad, bias_grad, atol=1e-6))

    def test_Conv2d(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 5, 6
        kernel_size = 3
        padding = 2
        stride = 2
        for _ in range(100):
            torch_layer = nn.Conv2d(
                n_in, n_out, kernel_size, padding=padding, stride=stride
            )
            custom_layer = Conv2d(
                n_in, n_out, kernel_size, padding=padding, stride=stride
            )
            self.set_weight_and_bias(custom_layer, torch_layer)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(
                -1,
                1,
                (
                    bs,
                    n_out,
                    1 + (h + 2 * padding - kernel_size) // stride,
                    1 + (w + 2 * padding - kernel_size) // stride,
                ),
            ).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.update_grad_input(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

            custom_layer.acc_grad_parameters(layer_input, next_layer_grad)
            weight_grad = custom_layer.grad_w
            bias_grad = custom_layer.grad_b
            torch_weight_grad = torch_layer.weight.grad.data.numpy()
            torch_bias_grad = torch_layer.bias.grad.data.numpy()

            self.assertTrue(np.allclose(torch_weight_grad, weight_grad, atol=1e-6))
            self.assertTrue(np.allclose(torch_bias_grad, bias_grad, atol=1e-6))

    def test_Conv2d1x7(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 5, 6
        kernel_size = (1, 7)
        padding = (0, 3)
        stride = 2
        for _ in range(100):
            torch_layer = nn.Conv2d(
                n_in, n_out, kernel_size, padding=padding, stride=stride
            )
            custom_layer = Conv2d(
                n_in, n_out, kernel_size, padding=padding, stride=stride
            )
            self.set_weight_and_bias(custom_layer, torch_layer)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(
                -1,
                1,
                (
                    bs,
                    n_out,
                    1 + (h + 2 * padding[0] - kernel_size[0]) // stride,
                    1 + (w + 2 * padding[1] - kernel_size[1]) // stride,
                ),
            ).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.update_grad_input(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

            custom_layer.acc_grad_parameters(layer_input, next_layer_grad)
            weight_grad = custom_layer.grad_w
            bias_grad = custom_layer.grad_b
            torch_weight_grad = torch_layer.weight.grad.data.numpy()
            torch_bias_grad = torch_layer.bias.grad.data.numpy()

            self.assertTrue(np.allclose(torch_weight_grad, weight_grad, atol=1e-6))
            self.assertTrue(np.allclose(torch_bias_grad, bias_grad, atol=1e-6))

    def test_Flatten(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in = 2, 3
        h, w = 4, 5

        for _ in range(100):
            torch_layer = nn.Flatten(1)
            custom_layer = Flatten(1)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(
                -1,
                1,
                (bs, n_in * h * w),
            ).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-6
                )
            )

            custom_layer_grad = custom_layer.update_grad_input(
                layer_input, next_layer_grad
            )
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-6
                )
            )

    def test_inceptionA(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 5, 6

        for _ in range(10):
            torch_layer = TorchInceptionA(n_in, n_out)
            custom_layer = InceptionA(n_in, n_out)

            for rule in [
                Rule("branch1x1", False),
                Rule("branch5x5", True, range(2)),
                Rule("branch3x3", True, range(3)),
                Rule("branch_pool", True, range(1, 2), add_id=False),
            ]:
                self.prepare_basic_conv(custom_layer, torch_layer, rule)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(-1, 1, (bs, 224 + n_out, h, w)).astype(
                np.float32
            )

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-5
                )
            )

            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-4
                )
            )

    def test_inceptionB(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 5, 6

        for _ in range(10):
            torch_layer = TorchInceptionB(n_in)
            custom_layer = InceptionB(n_in)

            for rule in [Rule("branch3x3", False), Rule("_branch3x3", True, range(3))]:
                self.prepare_basic_conv(custom_layer, torch_layer, rule)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(
                -1, 1, (bs, 480 + n_in, 1 + (h - 3) // 2, 1 + (w - 3) // 2)
            ).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-5
                )
            )

            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-4
                )
            )

    def test_inceptionC(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 5, 6
        channels_7x7 = 192

        for _ in range(10):
            torch_layer = TorchInceptionC(n_in, channels_7x7)
            custom_layer = InceptionC(n_in, channels_7x7)

            for rule in [
                Rule("branch1x1", False),
                Rule("branch7x7", True, range(3)),
                Rule("_branch7x7", True, range(5)),
                Rule("branch_pool", True, range(1, 2), add_id=False),
            ]:
                self.prepare_basic_conv(custom_layer, torch_layer, rule)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(-1, 1, (bs, 768, h, w)).astype(
                np.float32
            )

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-5
                )
            )

            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-4
                )
            )

    def test_inceptionD(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 5, 6

        for _ in range(10):
            torch_layer = TorchInceptionD(n_in)
            custom_layer = InceptionD(n_in)

            for rule in [
                Rule("branch3x3", True, range(2)),
                Rule("branch7x7x3", True, range(4)),
            ]:
                self.prepare_basic_conv(custom_layer, torch_layer, rule)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(
                -1, 1, (bs, 512 + n_in, 1 + (h - 3) // 2, 1 + (w - 3) // 2)
            ).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-4
                )
            )

            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-4
                )
            )

    def test_inceptionE(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 5, 6

        for _ in range(5):
            torch_layer = TorchInceptionE(n_in)
            custom_layer = InceptionE(n_in)

            for rule in [
                Rule("branch1x1", False),
                Rule("branch3x3_1", False),
                Rule("branch3x3_2a", False),
                Rule("branch3x3_2b", False),
                Rule("_branch3x3_1", False),
                Rule("_branch3x3_2", False),
                Rule("_branch3x3_3a", False),
                Rule("_branch3x3_3b", False),
                Rule("branch_pool", False),
            ]:
                self.prepare_basic_conv(custom_layer, torch_layer, rule)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(-1, 1, (bs, 2048, h, w)).astype(
                np.float32
            )

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-4
                )
            )

            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-4
                )
            )

    def test_InceptionAux(self):
        np.random.seed(21)
        torch.manual_seed(21)

        bs, n_in, n_out = 2, 3, 4
        h, w = 32, 64
        num_classes = 1000

        for _ in range(5):
            torch_layer = TorchInceptionAux(n_in, num_classes)
            custom_layer = InceptionAux(n_in, num_classes)

            self.prepare_conv_bn(custom_layer.inception_aux[1], torch_layer.conv1)
            self.prepare_conv_bn(custom_layer.inception_aux[2], torch_layer.conv2)
            self.set_weight_and_bias(custom_layer.inception_aux[5], torch_layer.fc)

            layer_input = np.random.uniform(-1, 1, (bs, n_in, h, w)).astype(np.float32)
            next_layer_grad = np.random.uniform(-1, 1, (bs, num_classes)).astype(np.float32)

            custom_layer_output = custom_layer.update_output(layer_input)
            layer_input_var = Variable(
                torch.from_numpy(layer_input), requires_grad=True
            )
            torch_layer_output_var = torch_layer(layer_input_var)
            self.assertTrue(
                np.allclose(
                    torch_layer_output_var.data.numpy(), custom_layer_output, atol=1e-4
                )
            )

            custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
            torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
            torch_layer_grad_var = layer_input_var.grad
            self.assertTrue(
                np.allclose(
                    torch_layer_grad_var.data.numpy(), custom_layer_grad, atol=1e-4
                )
            )
