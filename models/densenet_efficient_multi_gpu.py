from __future__ import print_function
import math
import torch
from torch.autograd import Function, Variable
from torch.nn import Parameter, Module
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from operator import mul
from functools import reduce


def create_multi_gpu_storage(size=1024):
    multi_storage = []
    device_cnt = torch.cuda.device_count()
    for device_no in range(device_cnt):
        with torch.cuda.device(device_no):
            multi_storage.append(torch.Storage(size).cuda())
    return multi_storage


class _SharedAllocation(object):
    def __init__(self, storage):
        self.multi_storage = storage
        self.storage = self.multi_storage[0]

    def type(self, t):
        self.storage = self.storage.type(t)

    def type_as(self, obj):
        new_sto = []
        if isinstance(obj, Variable):
            for sto in self.multi_storage:
                new_sto.append(sto.type(obj.data.storage().type()))
        elif isinstance(obj, torch._TensorBase):
            for sto in self.multi_storage:
                new_sto.append(sto.type(obj.storage().type()))
        else:
            for sto in self.multi_storage:
                new_sto.append(sto.type(obj.type()))
        self.multi_storage = new_sto

    def change_device(self, id):
        return self.multi_storage[id]

    def resize_(self, size):
        for device_no, sto in enumerate(self.multi_storage):
            if sto.size() < size:
                with torch.cuda.device(device_no): # this line is crucial!!
                    sto.resize_(size)
        return self


class EfficientDensenetBottleneck(Module):
    """
    A optimized layer which encapsulates the batch normalization, ReLU, and
    convolution operations within the bottleneck of a DenseNet layer.

    This layer usage shared memory allocations to store the outputs of the
    concatenation and batch normalization features. Because the shared memory
    is not perminant, these features are recomputed during the backward pass.
    """
    def __init__(self, shared_alloc, num_input_channels, num_output_channels):
        super(EfficientDensenetBottleneck, self).__init__()
        self.shared_alloc = shared_alloc
        self.num_input_channels = num_input_channels
        self.norm_weight = Parameter(torch.Tensor(num_input_channels))
        self.norm_bias = Parameter(torch.Tensor(num_input_channels))
        self.register_buffer('norm_running_mean', torch.zeros(num_input_channels))
        self.register_buffer('norm_running_var', torch.ones(num_input_channels))
        self.conv_weight = Parameter(torch.Tensor(num_output_channels, num_input_channels, 1, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        self.norm_running_mean.zero_()
        self.norm_running_var.fill_(1)
        self.norm_weight.data.uniform_()
        self.norm_bias.data.zero_()
        stdv = 1. / math.sqrt(self.num_input_channels * 1 * 1)
        self.conv_weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        if isinstance(inputs, Variable):
            inputs = [inputs]
        fn = _EfficientDensenetBottleneckFn(self.shared_alloc,
                                            self.norm_running_mean, self.norm_running_var,
                                            stride=1, padding=0, dilation=1, groups=1,
                                            training=self.training, momentum=0.1, eps=1e-5)
        return fn(self.norm_weight, self.norm_bias, self.conv_weight, *inputs)


class _DenseLayer(Module):
    def __init__(self, shared_alloc, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.shared_alloc = shared_alloc
        self.drop_rate = drop_rate
        self.bn_size = bn_size

        if bn_size > 0:
            self.efficient = EfficientDensenetBottleneck(shared_alloc,
                                                         num_input_features, bn_size * growth_rate)
            self.bn = nn.BatchNorm2d(bn_size * growth_rate)
            self.relu = nn.ReLU(inplace=True)
            self.conv = nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.efficient = EfficientDensenetBottleneck(shared_alloc,
                                                         num_input_features, growth_rate)
            self.conv1 = nn.Conv2d(num_input_features, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if isinstance(x, Variable):
            prev_features = [x]
        else:
            prev_features = x
        out = self.efficient(prev_features)
        # out = self.conv1(out)
        if self.bn_size > 0:
            out = self.bn(out)
            out = self.relu(out)
            out = self.conv(out)
        return out


class _DenseBlock(Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, storage):
        super(_DenseBlock, self).__init__()
        self.storage = storage
        self.final_num_features = num_input_features + (growth_rate * num_layers)
        self.shared_alloc = _SharedAllocation(storage)
        self.register_buffer('CatBN_output_buffer', self.storage)

        print('bnsize _DenseBlock', bn_size)

        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(self.shared_alloc,
                                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        # Update storage type
        self.shared_alloc.type_as(x)

        # Resize storage
        final_size = list(x.size())
        final_size[1] = self.final_num_features
        final_storage_size = reduce(mul, final_size, 1)
        self.shared_alloc.resize_(final_storage_size)

        outputs = [x]
        for module in self.children():
            outputs.append(module.forward(outputs))
        return torch.cat(outputs, dim=1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseNetEfficientMulti(Module):
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
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0., avgpool_size=8,
                 num_classes=10):

        super(DenseNetEfficientMulti, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between '
        self.avgpool_size = avgpool_size

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        # Each dense block
        storage = create_multi_gpu_storage()
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate, storage=storage)
            self.features.add_module('denseblock_%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionBlock(in_planes=num_features,
                                    out_planes=int(num_features * compression),
                                        dropRate=drop_rate)
                self.features.add_module('transition_%d' % (i + 1), trans)
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

class _EfficientCat(object):
    def __init__(self, storage):
        self.storage = storage

    def forward(self, *inputs):
        # Get size of new varible
        self.all_num_channels = [input.size(1) for input in inputs]
        size = list(inputs[0].size())
        for num_channels in self.all_num_channels[1:]:
            size[1] += num_channels

        # Create variable, using existing storage
        cur_device_id = inputs[0].get_device()
        res = type(inputs[0])(self.storage.change_device(cur_device_id)).resize_(size)

        assert inputs[0].get_device() == res.get_device(), \
            "input and output are not on the same chip!"
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


class _EfficientBatchNorm(object):
    def __init__(self, storage, running_mean, running_var,
            training=False, momentum=0.1, eps=1e-5):
        self.storage = storage
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, weight, bias, input):
        # Assert we're using cudnn
        for i in ([weight, bias, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use EfficientBatchNorm')

        # Create save variables
        self.save_mean = self.running_mean.new()
        self.save_mean.resize_as_(self.running_mean)
        self.save_var = self.running_var.new()
        self.save_var.resize_as_(self.running_var)

        # Do forward pass - store in input variable
        cur_device_id = weight.get_device()
        res = type(input)(self.storage.change_device(cur_device_id)).resize_as_(input)
        assert weight.get_device() == res.get_device(), \
            "input and output should be on the same chip!"

        torch._C._cudnn_batch_norm_forward(input, res,
                weight, bias,
                self.running_mean, self.running_var,
                self.save_mean, self.save_var,
                self.training,
                self.momentum,
                self.eps)
        return res

    def backward(self, weight, bias, input, grad_output):
        # Create grad variables
        grad_weight = weight.new()
        grad_weight.resize_as_(weight)
        grad_bias = bias.new()
        grad_bias.resize_as_(bias)

        # Run backwards pass - result stored in grad_output
        grad_input = grad_output
        torch._C._cudnn_batch_norm_backward(input, grad_output,
                grad_input, grad_weight, grad_bias,
                weight, self.running_mean, self.running_var,
                self.save_mean, self.save_var,
                self.training, self.eps)

        # Unpack grad_output
        res = tuple([grad_weight, grad_bias, grad_input])
        return res


class _EfficientReLU(object):
    def __init__(self):
        pass

    def forward(self, input):
        backend = torch._thnn.type2backend[type(input)]
        output = input
        backend.Threshold_updateOutput(
            backend.library_state,
            input,
            output,
            0,
            0,
            True
        )
        return output

    def backward(self, input, grad_output):
        grad_input = grad_output
        grad_input.masked_fill_(input <= 0, 0)
        return grad_input


class _EfficientConv2d(object):
    def __init__(self, stride=1, padding=0, dilation=1, groups=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def _output_size(self, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = self.padding
            kernel = self.dilation * (weight.size(d + 2) - 1) + 1
            stride = self.stride
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError("convolution input is too small (output would be {})".format(
                             'x'.join(map(str, output_size))))
        return output_size

    def forward(self, weight, bias, input):
        # Assert we're using cudnn
        for i in ([weight, bias, input]):
            if i is not None and not(cudnn.is_acceptable(i)):
                raise Exception('You must be using CUDNN to use _EfficientBatchNorm')

        res = input.new(*self._output_size(input, weight))
        self._cudnn_info = torch._C._cudnn_convolution_full_forward(
            input, weight, bias, res,
            (self.padding, self.padding),
            (self.stride, self.stride),
            (self.dilation, self.dilation),
            self.groups, cudnn.benchmark
        )

        return res

    def backward(self, weight, bias, input, grad_output):
        grad_input = input.new()
        grad_input.resize_as_(input)
        torch._C._cudnn_convolution_backward_data(
            grad_output, grad_input, weight, self._cudnn_info,
            cudnn.benchmark)

        grad_weight = weight.new().resize_as_(weight)
        torch._C._cudnn_convolution_backward_filter(grad_output, input, grad_weight, self._cudnn_info,
                                                    cudnn.benchmark)

        if bias is not None:
            grad_bias = bias.new().resize_as_(bias)
            torch._C._cudnn_convolution_backward_bias(grad_output, grad_bias, self._cudnn_info)
        else:
            grad_bias = None

        return grad_weight, grad_bias, grad_input



class _EfficientDensenetBottleneckFn(Function):
    """
    The autograd function which performs the efficient bottlenck operations.
    Each of the sub-operations -- concatenation, batch normalization, ReLU,
    and convolution -- are abstracted into their own classes
    """
    def __init__(self, shared_alloc,
                 running_mean, running_var,
                 stride=1, padding=0, dilation=1, groups=1,
                 training=False, momentum=0.1, eps=1e-5):
        super(_EfficientDensenetBottleneckFn, self).__init__()

        self.efficient_cat = _EfficientCat(shared_alloc)
        self.efficient_batch_norm = _EfficientBatchNorm(shared_alloc, running_mean,
                                                        running_var, training, momentum, eps)
        self.efficient_relu = _EfficientReLU()

        self.efficient_conv = _EfficientConv2d(stride, padding, dilation, groups)


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
        bias = None
        conv_output = self.efficient_conv.forward(conv_weight, None, relu_output)

        self.bn_weight = bn_weight
        self.bn_bias = bn_bias
        self.conv_weight = conv_weight
        self.inputs = inputs
        return conv_output

    def backward(self, grad_output):
        # Turn off bn training status, and temporarily reset statistics

        training = self.efficient_batch_norm.training
        self.curr_running_mean.copy_(self.efficient_batch_norm.running_mean)
        self.curr_running_var.copy_(self.efficient_batch_norm.running_var)
        # self.efficient_batch_norm.training = False
        self.efficient_batch_norm.running_mean.copy_(self.prev_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.prev_running_var)

        # Recompute concat and BN
        cat_output = self.efficient_cat.forward(*self.inputs)
        bn_output = self.efficient_batch_norm.forward(self.bn_weight, self.bn_bias, cat_output)
        relu_output = self.efficient_relu.forward(bn_output)

        # Conv backward
        conv_weight_grad, _, conv_grad_output = self.efficient_conv.backward(
                self.conv_weight, None, relu_output, grad_output)

        # ReLU backward
        relu_grad_output = self.efficient_relu.backward(bn_output, conv_grad_output)

        # BN backward
        cat_output = self.efficient_cat.forward(*self.inputs) # recompute cat_output because bn_output override the storage (L481)
                                                              # multi_gpu version is slightly different from the single gpu that
                                                              # we only use one shared_allocation for both BN and Cat
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)
        bn_weight_grad, bn_bias_grad, bn_grad_output = self.efficient_batch_norm.backward(
                self.bn_weight, self.bn_bias, cat_output, relu_grad_output)

        # Input backward
        grad_inputs = self.efficient_cat.backward(bn_grad_output)
        # Reset bn training status and statistics
        self.efficient_batch_norm.training = training
        self.efficient_batch_norm.running_mean.copy_(self.curr_running_mean)
        self.efficient_batch_norm.running_var.copy_(self.curr_running_var)

        return tuple([bn_weight_grad, bn_bias_grad, conv_weight_grad] + list(grad_inputs))
