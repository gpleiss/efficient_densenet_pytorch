# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# This code supports original DenseNet as well

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficient_densenet_bottleneck import EfficientDensenetBottleneck
from models.utils import Buffer
from operator import mul
from collections import OrderedDict
from torch.autograd import Variable, Function
from torchvision.models.densenet import _Transition
from torch.backends import cudnn


class _SharedConv2dFunction(Function):
    def __init__(self, buffer, buffer_start, in_channels, out_channels,
                 stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
        self.buffer = buffer
        self.buffer_start = buffer_start
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, input, weight):
        # Assert we're using cudnn
        for i in ([weight, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use EfficientBatchNorm')

        # get storage for output
        output = self.buffer[:, self.buffer_start:(self.buffer_start + self.out_channels)]
        # print(self.buffer.size(), output.size(), input.size(), weight.size())
        # print(self.buffer.type(), output.type(), input.type(), weight.type())
        # print(self.padding, self.stride, self.dilation, self.groups)
        # print(input.is_contiguous(), weight.is_contiguous(), output.is_contiguous())

        # Perform convolution
        self._cudnn_info = torch._C._cudnn_convolution_full_forward(
            input, weight, None, output,
            self.padding, self.stride, self.dilation,
            self.groups, cudnn.benchmark)

        self.save_for_backward(input, weight)
        return self.buffer[:, :(self.buffer_start + self.out_channels)]

    def backward(self, grad_output):
        input, weight = self.saved_tensors
        grad_input = None
        grad_weight = None

        if self.needs_input_grad[0]:
            grad_input = type(input)(self.buffer)
            grad_input.resize_as_(input)
            torch._C._cudnn_convolution_backward_data(
                grad_output, grad_input, weight, self._cudnn_info,
                cudnn.benchmark)

        if self.needs_input_grad[1]:
            grad_weight = weight.new().resize_as_(weight)
            torch._C._cudnn_convolution_backward_filter(
                grad_output, input, grad_weight, self._cudnn_info,
                cudnn.benchmark)

        return grad_input, grad_weight


class _SharedConv2d(nn.Conv2d):
    def __init__(self, buffer, buffer_start, in_channels, out_channels, kernel_size,
                 stride=1, padding=1):
        super(_SharedConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                            stride, padding, bias=False)
        self.buffer = buffer
        self.buffer_start = buffer_start

    def forward(self, input):
        buffer = type(input.data)(self.buffer.storage).view(input.size(0), -1, input.size(2), input.size(3))
        function = _SharedConv2dFunction(buffer, self.buffer_start, self.in_channels,
                                         self.out_channels, self.stride,
                                         self.padding)
        return function(input, self.weight)


class _DenseLayer(nn.Sequential):
    def __init__(self, buffr_1, buffr_2, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.buffr_1 = buffr_1
        self.buffr_2 = buffr_2
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate

        self.add_module('bn', EfficientDensenetBottleneck(buffr_1,
            num_input_features, bn_size * growth_rate))
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu.2', nn.ReLU(inplace=True))
        self.add_module('conv.2', _SharedConv2d(self.buffr_2, self.num_input_features,
            bn_size * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        result = super(_DenseLayer, self).forward(input)
        if self.drop_rate > 0:
            new_features = result[:, self.num_input_features:(self.num_input_features + self.growth_rate)]
            F.dropout(new_features, p=self.drop_rate, training=self.training, inplace=True)
        return result


class _DenseBlock(nn.Container):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, storage_size=1024):
        input_storage_1 = torch.Storage(storage_size)
        input_storage_2 = torch.Storage(storage_size)
        self.final_num_features = num_input_features + (growth_rate * num_layers)
        self.buffr_1 = Buffer(input_storage_1)
        self.buffr_2 = Buffer(input_storage_2)

        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(self.buffr_1, self.buffr_2,
                    num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


    def forward(self, x):
        # Update storage type
        self.buffr_1.type_as(x)
        self.buffr_2.type_as(x)

        # Resize storage
        final_size = list(x.size())
        final_size[1] = self.final_num_features
        final_storage_size = reduce(mul, final_size, 1)
        self.buffr_1.resize_(final_storage_size)
        self.buffr_2.resize_(final_storage_size)

        # Copy over stuff to buffer 2
        tensor = type(x.data)(self.buffr_2.storage)
        input_size = x.size()
        tensor = tensor.view(x.size(0), -1, x.size(2), x.size(3))
        tensor[:, 0:x.size(1)].copy_(x.data)

        for module in self.children():
            x = module(x)
        return x

class DenseNetEfficient(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=8,
                 num_classes=10):

        super(DenseNetEfficient, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between '
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

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
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out
