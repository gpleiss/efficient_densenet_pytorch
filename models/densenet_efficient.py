# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# This code supports original DenseNet as well

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from operator import mul
from collections import OrderedDict
from torch.autograd import Function, Variable
from torch.nn.modules.batchnorm import _BatchNorm
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


class _EfficientBatchNorm(_EfficientCatFn):
    def __init__(self, storage, running_mean, running_var,
            training=False, momentum=0.1, eps=1e-5):
        self.storage = storage
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps


    def forward(self, weight, bias, *inputs):
        # Assert we're using cudnn
        for input in ([weight, bias] + list(inputs)):
            if not(cudnn.is_acceptable(input)):
                raise Exception('You must be using CUDNN to use EfficientBatchNorm')

        # Create save variables if they don't exist
        self.save_mean = self.save_mean if hasattr(self, 'save_mean') else self.running_mean.new()
        self.save_mean.resize_as_(self.running_mean)
        self.save_var = self.save_var if hasattr(self, 'save_var') else self.running_var.new()
        self.save_var.resize_as_(self.running_var)

        # Buffer for weights and bias
        self.inputs = inputs
        self.weight = weight
        self.bias = bias

        # Do forward pass
        res = super(_EfficientBatchNorm, self).forward(*inputs)
        torch._C._cudnn_batch_norm_forward(res, res,
                weight, bias,
                self.running_mean, self.running_var,
                self.save_mean, self.save_var,
                self.training, self.momentum, self.eps)

        return res


    def backward(self, grad_output):
        # Create grad variables if they don't exist
        if not hasattr(self, 'grad_weight'):
            self.grad_weight = self.weight.new()
            self.grad_weight.resize_as_(self.weight)
        if not hasattr(self, 'grad_bias'):
            self.grad_bias = self.bias.new()
            self.grad_bias.resize_as_(self.bias)

        # Get input through a forward pass
        input = self.forward(self.weight, self.bias, *self.inputs)

        # Run backwards pass - result stored in grad_output
        torch._C._cudnn_batch_norm_backward(input, grad_output,
                grad_output, self.grad_weight, self.grad_bias,
                self.weight, self.running_mean, self.running_var,
                self.save_mean, self.save_var,
                self.training, self.eps)

        # Unpack grad_output
        unpacked_grad_input = super(_EfficientBatchNorm, self).backward(grad_output)
        res = tuple([self.grad_weight, self.grad_bias] + list(unpacked_grad_input))
        return res


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


    def batch_norm(self, inputs, running_mean, running_var, weight=None, bias=None,
            training=False, momentum=0.1, eps=1e-5):
        func = _EfficientBatchNorm(self.storage, running_mean, running_var, training, momentum, eps)
        res = func(weight, bias, *inputs)
        return res


class _EfficientBatchNorm2d(_BatchNorm):
    def __init__(self, buffr, *args, **kwargs):
        self.buffr = buffr
        return super(_EfficientBatchNorm2d, self).__init__(*args, **kwargs)


    def _check_input_dim(self, inputs):
	for input in inputs:
	    if input.dim() != 4:
		raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
	nchannels = sum([input.size(1) for input in inputs])
        if nchannels != self.running_mean.nelement():
            raise ValueError('got {}-feature tensors, expected {}'.format(nchannels, self.num_features))


    def forward(self, inputs):
	self._check_input_dim(inputs)
	return self.buffr.batch_norm(inputs, self.running_mean, self.running_var,
	    self.weight, self.bias, self.training, self.momentum, self.eps)


class _DenseLayer(nn.Sequential):
    def __init__(self, buffr, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.buffr = buffr
        self.drop_rate = drop_rate

        self.add_module('norm.1', _EfficientBatchNorm2d(self.buffr, num_input_features)),
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


    def forward(self, x):
        # if isinstance(x, Variable):
            # prev_features = x
        # else:
            # prev_features = self.buffr.cat(x)
        new_features = super(_DenseLayer, self).forward(x)
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
        return torch.cat(outputs, dim=1)


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
