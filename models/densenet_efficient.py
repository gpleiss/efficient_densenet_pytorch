# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# This code supports original DenseNet as well

import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import mul
from collections import OrderedDict
from torch.autograd import Function, Variable
from torchvision.models.densenet import _Transition


class _EfficientCatFn(Function):
    def __init__(self, storage):
        self.storage = storage


    def forward(self, *inputs):
        # Get size of new varible
        self.all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in self.all_num_channels[1:]:
            size[1] += num_channels

        # Create variable, using existing storage
        res = type(inputs[0])(self.storage).resize_(size)
        torch.cat(inputs, dim=1, out=res)
        return res


    def backward(self, grad_output):
        # Return a table of tensors pointing to same storage
        res = []
        index = 0
        for num_channels in self.all_num_channels:
            new_index = num_channels + index
            res.append(grad_output[:, index:new_index])
            index = new_index

        return tuple(res)


class _Buffer(object):
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


    def cat(self, input_vars):
        res = _EfficientCatFn(self.storage)(*input_vars)
        return res


class _DenseLayer(nn.Sequential):
    def __init__(self, buffr, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        if bn_size > 0:
            self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                            growth_rate, kernel_size=1, stride=1, bias=False)),
            self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu.2', nn.ReLU(inplace=True)),
            self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)),
        else:
            self.add_module('conv.1', nn.Conv2d(num_input_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.buffr = buffr


    def forward(self, x):
        if isinstance(x, Variable):
            prev_features = x
        else:
            prev_features = self.buffr.cat(x)
        new_features = super(_DenseLayer, self).forward(prev_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Container):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, storage_size=1024):
        input_storage = torch.Storage(storage_size)
        self.final_num_features = num_input_features + (growth_rate * num_layers)
        self.buffr = _Buffer(input_storage)

        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(self.buffr, num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


    def forward(self, x):
        # Update storage type
        self.buffr.type_as(x)

        # Resize storage
        final_size = list(x.size())
        final_size[1] = self.final_num_features
        final_storage_size = reduce(mul, final_size, 1)
        self.buffr.resize_(final_storage_size)

        outputs = [x]
        for module in self.children():
            outputs.append(module.forward(outputs))
        return self.buffr.cat(outputs)


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
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
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
