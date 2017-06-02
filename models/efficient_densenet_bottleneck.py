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
