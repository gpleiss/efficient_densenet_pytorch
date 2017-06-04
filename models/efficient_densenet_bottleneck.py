import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Function, Variable
from torch.nn import Parameter

from models.utils import EfficientCat, EfficientBatchNorm, EfficientReLU, EfficientConv2d

class _EfficientDensenetBottleneckFn(Function):
    def __init__(self, buffr_1, buffr_2,
            running_mean, running_var,
            stride=1, padding=0, dilation=1, groups=1,
            training=False, momentum=0.1, eps=1e-5):

        self.efficient_cat = EfficientCat(buffr_1.storage)
        self.efficient_batch_norm = EfficientBatchNorm(buffr_1.storage, running_mean, running_var,
                training, momentum, eps)
        self.efficient_relu = EfficientReLU()
        self.efficient_conv = EfficientConv2d(buffr_2.storage, stride, padding, dilation, groups)

        # Buffers to store old versions of bn statistics
        self.prev_running_mean = self.efficient_batch_norm.running_mean.new()
        self.prev_running_mean.resize_as_(self.efficient_batch_norm.running_mean)
        self.prev_running_var = self.efficient_batch_norm.running_var.new()
        self.prev_running_var.resize_as_(self.efficient_batch_norm.running_var)
        self.curr_running_mean = self.efficient_batch_norm.running_mean.new()
        self.curr_running_mean.resize_as_(self.efficient_batch_norm.running_mean)
        self.curr_running_var = self.efficient_batch_norm.running_var.new()
        self.curr_running_var.resize_as_(self.efficient_batch_norm.running_var)


    def forward(self, bn_weight, bn_bias, conv_weight, *inputs):
        self.prev_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.prev_running_var.copy_(self.efficient_batch_norm.running_var)

        bn_input = self.efficient_cat.forward(*inputs)
        bn_output = self.efficient_batch_norm.forward(bn_weight, bn_bias, bn_input)
        relu_output = self.efficient_relu.forward(bn_output)
        conv_output = self.efficient_conv.forward(conv_weight, None, relu_output)

        self.bn_weight = bn_weight
        self.bn_bias = bn_bias
        self.conv_weight = conv_weight
        self.inputs = inputs
        return conv_output


    def backward(self, grad_output):
        # Turn off bn training status, and temporarely reset statistics
        training = self.efficient_batch_norm.training
        self.curr_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.curr_running_var.copy_(self.efficient_batch_norm.running_var)
        # self.efficient_batch_norm.training = False
        self.efficient_batch_norm.running_mean.copy_(self.prev_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.prev_running_var)

        # Recompute BN
        bn_input = self.efficient_cat.forward(*self.inputs)
        bn_output = self.efficient_batch_norm.forward(self.bn_weight, self.bn_bias, bn_input)
        relu_output = self.efficient_relu.forward(bn_output)

        # Conv backward
        conv_weight_grad, _, conv_grad_output = self.efficient_conv.backward(
                self.conv_weight, None, relu_output, grad_output)

        # ReLU backward
        relu_grad_output = self.efficient_relu.backward(bn_output, conv_grad_output)

        # BN backward
        bn_input = self.efficient_cat.forward(*self.inputs)
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)
        bn_weight_grad, bn_bias_grad, bn_grad_output = self.efficient_batch_norm.backward(
                self.bn_weight, self.bn_bias, bn_input, relu_grad_output)

        # Input backward
        grad_inputs = self.efficient_cat.backward(bn_grad_output)

        # Reset bn training status and statistics
        self.efficient_batch_norm.training = training
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)

        return tuple([bn_weight_grad, bn_bias_grad, conv_weight_grad] + list(grad_inputs))


class EfficientDensenetBottleneck(nn.Module):
    def __init__(self, buffr_1, buffr_2, num_input_channels, num_output_channels):

        super(EfficientDensenetBottleneck, self).__init__()
        self.buffr_1 = buffr_1
        self.buffr_2 = buffr_2
        self.num_input_channels = num_input_channels

        self.norm_weight = Parameter(torch.Tensor(num_input_channels))
        self.norm_bias = Parameter(torch.Tensor(num_input_channels))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.conv_weight = Parameter(torch.Tensor(num_output_channels, num_input_channels, 3, 3))
        self._reset_parameters()


    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels * 3 * 3)
        self.conv_weight.data.uniform_(-stdv, stdv)


    def forward(self, inputs):
        if isinstance(inputs, Variable):
            inputs = [inputs]
        fn = _EfficientDensenetBottleneckFn(self.buffr_1, self.buffr_2,
            self.norm_running_mean, self.norm_running_var,
            stride=1, padding=1, dilation=1, groups=1,
            training=self.training, momentum=0.1, eps=1e-5)
        return fn(self.norm_weight, self.norm_bias, self.conv_weight, *inputs)
