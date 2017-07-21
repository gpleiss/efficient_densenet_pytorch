import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from collections import OrderedDict

from models.densenet_efficient import _EfficientDensenetBottleneck, _SharedAllocation


def almost_equal(self, other, eps=1e-5):
    return torch.max((self - other).abs()) <= eps


def test_forward_training_false_computes_forward_pass():
    bn_weight = torch.randn(8).cuda()
    bn_bias = torch.randn(8).cuda()
    bn_running_mean = torch.randn(8).cuda()
    bn_running_var = torch.randn(8).abs().cuda()
    conv_weight = torch.randn(4, 8, 3, 3).cuda()
    input_1 = torch.randn(4, 6, 4, 4).cuda()
    input_2 = torch.randn(4, 2, 4, 4).cuda()

    layer = nn.Sequential(OrderedDict([
        ('norm', nn.BatchNorm2d(8)),
        ('relu', nn.ReLU(inplace=True)),
        ('conv', nn.Conv2d(8, 4, bias=None, kernel_size=3, stride=1, padding=1)),
    ])).cuda()
    layer.eval()
    layer.norm.weight.data.copy_(bn_weight)
    layer.norm.bias.data.copy_(bn_bias)
    layer.norm.running_mean.copy_(bn_running_mean)
    layer.norm.running_var.copy_(bn_running_var)
    layer.conv.weight.data.copy_(conv_weight)

    input_1_var = Variable(input_1)
    input_2_var = Variable(input_2)
    out_var = layer(torch.cat([input_1_var, input_2_var], dim=1))

    storage_1 = torch.Storage(4 * 8 * 3 * 3).cuda()
    storage_2 = torch.Storage(4 * 8 * 3 * 3).cuda()
    layer_efficient = _EfficientDensenetBottleneck(
        _SharedAllocation(storage_1), _SharedAllocation(storage_2), 8, 4
    ).cuda()
    layer_efficient.eval()
    layer_efficient.norm_weight.data.copy_(bn_weight)
    layer_efficient.norm_bias.data.copy_(bn_bias)
    layer_efficient.norm_running_mean.copy_(bn_running_mean)
    layer_efficient.norm_running_var.copy_(bn_running_var)
    layer_efficient.conv_weight.data.copy_(conv_weight)

    input_efficient_1_var = Variable(input_1)
    input_efficient_2_var = Variable(input_2)
    out_efficient_var = layer_efficient([input_efficient_1_var, input_efficient_2_var])

    assert(almost_equal(out_var.data, out_efficient_var.data))


def test_forward_training_false_computes_forward_pass():
    bn_weight = torch.randn(8).cuda()
    bn_bias = torch.randn(8).cuda()
    bn_running_mean = torch.randn(8).cuda()
    bn_running_var = torch.randn(8).abs().cuda()
    conv_weight = torch.randn(4, 8, 1, 1).cuda()
    input_1 = torch.randn(4, 6, 4, 4).cuda()
    input_2 = torch.randn(4, 2, 4, 4).cuda()

    layer = nn.Sequential(OrderedDict([
        ('norm', nn.BatchNorm2d(8)),
        ('relu', nn.ReLU(inplace=True)),
        ('conv', nn.Conv2d(8, 4, bias=None, kernel_size=1, stride=1)),
    ])).cuda()
    layer.train()
    layer.norm.weight.data.copy_(bn_weight)
    layer.norm.bias.data.copy_(bn_bias)
    layer.norm.running_mean.copy_(bn_running_mean)
    layer.norm.running_var.copy_(bn_running_var)
    layer.conv.weight.data.copy_(conv_weight)

    input_1_var = Variable(input_1)
    input_2_var = Variable(input_2)
    out_var = layer(torch.cat([input_1_var, input_2_var], dim=1))

    storage_1 = torch.Storage(4 * 8 * 3 * 3).cuda()
    storage_2 = torch.Storage(4 * 8 * 3 * 3).cuda()
    layer_efficient = _EfficientDensenetBottleneck(
        _SharedAllocation(storage_1), _SharedAllocation(storage_2), 8, 4
    ).cuda()
    layer_efficient.train()
    layer_efficient.norm_weight.data.copy_(bn_weight)
    layer_efficient.norm_bias.data.copy_(bn_bias)
    layer_efficient.norm_running_mean.copy_(bn_running_mean)
    layer_efficient.norm_running_var.copy_(bn_running_var)
    layer_efficient.conv_weight.data.copy_(conv_weight)

    input_efficient_1_var = Variable(input_1)
    input_efficient_2_var = Variable(input_2)
    out_efficient_var = layer_efficient([input_efficient_1_var, input_efficient_2_var])

    assert(almost_equal(out_var.data, out_efficient_var.data))
    assert(almost_equal(layer.norm.running_mean, layer_efficient.norm_running_mean))
    assert(almost_equal(layer.norm.running_var, layer_efficient.norm_running_var))


def test_backward_computes_backward_pass():
    bn_weight = torch.randn(8).cuda()
    bn_bias = torch.randn(8).cuda()
    bn_running_mean = torch.randn(8).cuda()
    bn_running_var = torch.randn(8).abs().cuda()
    conv_weight = torch.randn(4, 8, 1, 1).cuda()
    input_1 = torch.randn(4, 6, 4, 4).cuda()
    input_2 = torch.randn(4, 2, 4, 4).cuda()

    layer = nn.Sequential(OrderedDict([
        ('norm', nn.BatchNorm2d(8)),
        ('relu', nn.ReLU(inplace=True)),
        ('conv', nn.Conv2d(8, 4, bias=None, kernel_size=1, stride=1)),
    ])).cuda()
    layer.train()
    layer.norm.weight.data.copy_(bn_weight)
    layer.norm.bias.data.copy_(bn_bias)
    layer.norm.running_mean.copy_(bn_running_mean)
    layer.norm.running_var.copy_(bn_running_var)
    layer.conv.weight.data.copy_(conv_weight)

    input_1_var = Variable(input_1, requires_grad=True)
    input_2_var = Variable(input_2, requires_grad=True)
    out_var = layer(torch.cat([input_1_var, input_2_var], dim=1))
    out_var.sum().backward()

    storage_1 = torch.Storage(4 * 8 * 3 * 3).cuda()
    storage_2 = torch.Storage(4 * 8 * 3 * 3).cuda()
    layer_efficient = _EfficientDensenetBottleneck(
        _SharedAllocation(storage_1), _SharedAllocation(storage_2), 8, 4
    ).cuda()
    layer_efficient.train()
    layer_efficient.norm_weight.data.copy_(bn_weight)
    layer_efficient.norm_bias.data.copy_(bn_bias)
    layer_efficient.norm_running_mean.copy_(bn_running_mean)
    layer_efficient.norm_running_var.copy_(bn_running_var)
    layer_efficient.conv_weight.data.copy_(conv_weight)

    input_efficient_1_var = Variable(input_1, requires_grad=True)
    input_efficient_2_var = Variable(input_2, requires_grad=True)
    out_efficient_var = layer_efficient([input_efficient_1_var, input_efficient_2_var])
    out_efficient_var.sum().backward()

    # print(input_1_var.grad.data[:, 0], input_efficient_1_var.grad.data[:, 0])
    assert(almost_equal(out_var.data, out_efficient_var.data))
    assert(almost_equal(layer.norm.running_mean, layer_efficient.norm_running_mean))
    assert(almost_equal(layer.norm.running_var, layer_efficient.norm_running_var))
    assert(almost_equal(layer.conv.weight.grad.data, layer_efficient.conv_weight.grad.data))
    assert(almost_equal(layer.norm.weight.grad.data, layer_efficient.norm_weight.grad.data))
    assert(almost_equal(layer.norm.bias.grad.data, layer_efficient.norm_bias.grad.data))
    assert(almost_equal(input_1_var.grad.data, input_efficient_1_var.grad.data))
    assert(almost_equal(input_2_var.grad.data, input_efficient_2_var.grad.data))
