import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class EfficientConv2d(object):
    def __init__(self, storage, stride=1, padding=0, dilation=1, groups=1):
        self.storage = storage
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
                raise Exception('You must be using CUDNN to use EfficientBatchNorm')

        res = input.new(*self._output_size(input, weight))
        self._cudnn_info = torch._C._cudnn_convolution_full_forward(
            input, weight, bias, res,
            (self.padding, self.padding),
            (self.stride, self.stride),
            (self.dilation, self.dilation),
            self.groups, cudnn.benchmark)

        return res


    def backward(self, weight, bias, input, grad_output):
        grad_input = type(input)(self.storage)
        grad_input.resize_as_(input)
	torch._C._cudnn_convolution_backward_data(
	    grad_output, grad_input, weight, self._cudnn_info,
	    cudnn.benchmark)

	grad_weight = weight.new().resize_as_(weight)
	torch._C._cudnn_convolution_backward_filter(
	    grad_output, input, grad_weight, self._cudnn_info,
	    cudnn.benchmark)

	if bias is not None:
	    grad_bias = bias.new().resize_as_(bias)
	    torch._C._cudnn_convolution_backward_bias(grad_output, grad_bias, self._cudnn_info)
	else:
	    grad_bias = None

        return grad_weight, grad_bias, grad_input
