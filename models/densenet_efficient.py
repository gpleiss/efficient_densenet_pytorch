# This implementation is a new efficient implementation of Densenet-BC,
# as described in "Memory-Efficient Implementation of DenseNets"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
from collections import OrderedDict
from torch.autograd import Variable, Function
from torch._thnn import type2backend
from torch.backends import cudnn


# I'm throwing all the gross code at the end of the file :)
# Let's start with the nice (and interesting) stuff


class _SharedAllocation(object):
    """
    A helper class which maintains a shared memory allocation.
    Used for concatenation and batch normalization.
    """
    def __init__(self, storage):
        self.storage = storage

    def type(self, t):
        self.storage = self.storage.type(t)

    def type_as(self, obj):
        if isinstance(obj, Variable):
            self.storage = self.storage.type(obj.data.storage().type())
        elif isinstance(obj, torch._TensorBase):
            self.storage = self.storage.type(obj.storage().type())
        else:
            self.storage = self.storage.type(obj.type())

    def resize_(self, size):
        if self.storage.size() < size:
            self.storage.resize_(size)
        return self


class _EfficientDensenetBottleneck(nn.Module):
    """
    A optimized layer which encapsulates the batch normalization, ReLU, and
    convolution operations within the bottleneck of a DenseNet layer.

    This layer usage shared memory allocations to store the outputs of the
    concatenation and batch normalization features. Because the shared memory
    is not perminant, these features are recomputed during the backward pass.
    """
    def __init__(self, shared_allocation_1, shared_allocation_2, num_input_channels, num_output_channels):
        super(_EfficientDensenetBottleneck, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.num_input_channels = num_input_channels

        self.norm_weight = nn.Parameter(torch.Tensor(num_input_channels))
        self.norm_bias = nn.Parameter(torch.Tensor(num_input_channels))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.conv_weight = nn.Parameter(torch.Tensor(num_output_channels, num_input_channels, 1, 1))
        self._reset_parameters()


    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels)
        self.conv_weight.data.uniform_(-stdv, stdv)


    def forward(self, inputs):
        if isinstance(inputs, Variable):
            inputs = [inputs]
        fn = _EfficientDensenetBottleneckFn(self.shared_allocation_1, self.shared_allocation_2,
                                            self.norm_running_mean, self.norm_running_var,
                                            stride=1, padding=0, dilation=1, groups=1,
                                            training=self.training, momentum=0.1, eps=1e-5)
        return fn(self.norm_weight, self.norm_bias, self.conv_weight, *inputs)


class _DenseLayer(nn.Sequential):
    def __init__(self, shared_allocation_1, shared_allocation_2,
                 num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.drop_rate = drop_rate

        self.add_module('bn', _EfficientDensenetBottleneck(shared_allocation_1, shared_allocation_2,
                                                           num_input_features, bn_size * growth_rate))
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),

    def forward(self, x):
        if isinstance(x, Variable):
            prev_features = [x]
        else:
            prev_features = x
        new_features = super(_DenseLayer, self).forward(prev_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Container):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, storage_size=1024):
        input_storage_1 = torch.Storage(storage_size)
        input_storage_2 = torch.Storage(storage_size)
        self.final_num_features = num_input_features + (growth_rate * num_layers)
        self.shared_allocation_1 = _SharedAllocation(input_storage_1)
        self.shared_allocation_2 = _SharedAllocation(input_storage_2)

        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(self.shared_allocation_1, self.shared_allocation_2,
                                num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


    def forward(self, x):
        # Update storage type
        self.shared_allocation_1.type_as(x)
        self.shared_allocation_2.type_as(x)

        # Resize storage
        final_size = list(x.size())
        final_size[1] = self.final_num_features
        final_storage_size = reduce(mul, final_size, 1)
        self.shared_allocation_1.resize_(final_storage_size)
        self.shared_allocation_2.resize_(final_storage_size)

        outputs = [x]
        for module in self.children():
            outputs.append(module.forward(outputs))
        return torch.cat(outputs, dim=1)


class DenseNetEfficient(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`

    This model uses shared memory allocations for the outputs of batch norm and
    concat operations, as described in `"Memory-Efficient Implementation of DenseNets"`.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True):

        super(DenseNetEfficient, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.avgpool_size = 8 if small_inputs else 7

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))


        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features
                                                            * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out


# Begin gross code :/
# Here's where we define the internals of the efficient bottleneck layer


class _EfficientDensenetBottleneckFn(Function):
    """
    The autograd function which performs the efficient bottlenck operations.
    Each of the sub-operations -- concatenation, batch normalization, ReLU,
    and convolution -- are abstracted into their own classes
    """
    def __init__(self, shared_allocation_1, shared_allocation_2,
                 running_mean, running_var,
                 stride=1, padding=0, dilation=1, groups=1,
                 training=False, momentum=0.1, eps=1e-5):

        self.shared_allocation_1 = shared_allocation_1
        self.shared_allocation_2 = shared_allocation_2
        self.running_mean = running_mean
        self.running_var = running_var
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.training = training
        self.momentum = momentum
        self.eps = eps

        # Buffers to store old versions of bn statistics
        self.prev_running_mean = self.running_mean.new(self.running_mean.size())
        self.prev_running_var = self.running_var.new(self.running_var.size())
        self.curr_running_mean = self.running_mean.new(self.running_mean.size())
        self.curr_running_var = self.running_var.new(self.running_var.size())

    def forward(self, bn_weight, bn_bias, conv_weight, *inputs):
        self.prev_running_mean.copy_(self.running_mean)
        self.prev_running_var.copy_(self.running_var)

        # Get size of new varible
        all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in all_num_channels[1:]:
            size[1] += num_channels

        # Create variable, using existing storage
        bn_input = type(inputs[0])(self.shared_allocation_1.storage).resize_(size)
        torch.cat(inputs, dim=1, out=bn_input)

        # Do batch norm and relu, but don't save the intermediate variables
        bn_output = F.batch_norm(Variable(bn_input, volatile=True), self.running_mean, self.running_var,
                                 Variable(bn_weight, volatile=True), Variable(bn_bias, volatile=True),
                                 training=self.training, momentum=self.momentum, eps=self.eps)
        relu_output = F.relu(bn_output, inplace=True)

        # Move the output of the ReLU to the shared allocation
        conv_input = type(inputs[0])(self.shared_allocation_2.storage).resize_(relu_output.size())
        conv_input.copy_(relu_output.data)
        relu_output.data.resize_(1)

        # Do convolution - and save variables because we'll need them for backward pass
        conv_input_var = Variable(conv_input, requires_grad=any(self.needs_input_grad))
        # conv_input_var = Variable(relu_output.data, requires_grad=any(self.needs_input_grad))
        conv_weight_var = Variable(conv_weight, requires_grad=self.needs_input_grad[2])
        conv_output = F.conv2d(conv_input_var, conv_weight_var, bias=None, stride=self.stride,
                               padding=0, dilation=1, groups=1)

        self.save_for_backward(bn_weight, bn_bias, conv_weight, *inputs)
        if any(self.needs_input_grad):
            self.conv_input_var = conv_input_var
            self.conv_weight_var = conv_weight_var
            self.conv_output = conv_output
        return conv_output.data


    def backward(self, grad_output):
        bn_weight, bn_bias, conv_weight = self.saved_tensors[:3]
        inputs = self.saved_tensors[3:]
        grads = [None] * len(self.saved_tensors)

        if not any(self.needs_input_grad):
            return grads

        # Conv backward
        self.conv_output.backward(gradient=grad_output)
        if self.needs_input_grad[2]:
            grads[2] = self.conv_weight_var.grad.data

        if any(self.needs_input_grad[:2]) or any(self.needs_input_grad[3:]):
            # Temporarily reset batch norm statistics
            self.curr_running_mean.copy_(self.running_mean)
            self.curr_running_var.copy_(self.running_var)
            self.running_mean.copy_(self.prev_running_mean)
            self.running_var.copy_(self.prev_running_var)

            # Recompute concat and BN
            all_num_channels = [input.size(1) for input in inputs]
            size = list(inputs[0].size())
            for num_channels in all_num_channels[1:]:
                size[1] += num_channels

            bn_input = Variable(type(inputs[0])(self.shared_allocation_1.storage),
                                requires_grad=any(self.needs_input_grad[3:]))
            bn_input.data.resize_(size)
            bn_weight_var = Variable(bn_weight, requires_grad=self.needs_input_grad[0])
            bn_bias_var = Variable(bn_bias, requires_grad=self.needs_input_grad[1])

            torch.cat(inputs, dim=1, out=bn_input.data)
            bn_output = F.batch_norm(bn_input, self.running_mean, self.running_var,
                                     bn_weight_var, bn_bias_var,
                                     training=self.training, momentum=self.momentum, eps=self.eps)
            relu_output = F.relu(bn_output, inplace=True)

            # ReLU/BN backward
            relu_output.backward(gradient=self.conv_input_var.grad)
            if self.needs_input_grad[0]:
                grads[0] = bn_weight_var.grad.data
            if self.needs_input_grad[1]:
                grads[1] = bn_bias_var.grad.data

            # Input grad
            if any(self.needs_input_grad[3:]):
                index = 0
                for i, num_channels in enumerate(all_num_channels):
                    new_index = num_channels + index
                    grads[3 + i] = bn_input.grad.data[:, index:new_index]
                    index = new_index

            # Reset bn training status and statistics
            self.running_mean.copy_(self.curr_running_mean)
            self.running_var.copy_(self.curr_running_var)

        return tuple(grads)
