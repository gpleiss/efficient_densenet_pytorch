import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Function, Variable
from torch.nn.modules.batchnorm import _BatchNorm

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
        input = super(_EfficientBatchNorm, self).forward(*self.inputs)

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
