import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class EfficientBatchNorm(object):
    def __init__(self, running_mean, running_var,
            training=False, momentum=0.1, eps=1e-5):
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
        res = input
        torch._C._cudnn_batch_norm_forward(input, res,
                weight, bias,
                self.running_mean, self.running_var,
                self.save_mean, self.save_var,
                self.training, self.momentum, self.eps)

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
