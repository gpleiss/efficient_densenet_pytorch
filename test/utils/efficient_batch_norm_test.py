import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from models.densenet_efficient import _EfficientBatchNorm


def almost_equal(self, other, eps=1e-5):
    return torch.max((self - other).abs()) <= eps


def test_forward_eval_mode_computes_forward_pass():
    momentum = 0.1
    eps = 1e-5

    weight = torch.randn(10).cuda()
    bias = torch.randn(10).cuda()
    running_mean = torch.randn(10).cuda()
    running_var = torch.randn(10).abs().cuda()

    input_1 = torch.randn(4, 5).cuda()
    input_2 = torch.randn(4, 5).cuda()
    storage = torch.Storage(40).cuda()

    bn = F.batch_norm(
        input=Variable(torch.cat([input_1, input_2], dim=1)),
        running_mean=running_mean,
        running_var=running_var,
        weight=Parameter(weight),
        bias=Parameter(bias),
        training=False,
        momentum=momentum,
        eps=eps
    ).data

    input_efficient = torch.cat([input_1, input_2], dim=1)
    func = _EfficientBatchNorm(
        storage=storage,
        running_mean=running_mean,
        running_var=running_var,
        training=False,
        momentum=momentum,
        eps=eps
    )
    bn_efficient = func.forward(weight, bias, input_efficient)

    assert(almost_equal(bn, bn_efficient))
    assert(bn_efficient.storage().data_ptr() == storage.data_ptr())


def test_forward_train_mode_computes_forward_pass():
    momentum = 0.1
    eps = 1e-5

    weight = torch.randn(10).cuda()
    bias = torch.randn(10).cuda()
    running_mean = torch.randn(10).cuda()
    running_var = torch.randn(10).abs().cuda()
    running_mean_efficient = running_mean.clone()
    running_var_efficient = running_var.clone()

    input_1 = torch.randn(4, 5).cuda()
    input_2 = torch.randn(4, 5).cuda()
    storage = torch.Storage(40).cuda()

    bn = F.batch_norm(
        input=Variable(torch.cat([input_1, input_2], dim=1)),
        running_mean=running_mean,
        running_var=running_var,
        weight=Parameter(weight),
        bias=Parameter(bias),
        training=True,
        momentum=momentum,
        eps=eps
    ).data

    input_efficient = torch.cat([input_1, input_2], dim=1)
    func = _EfficientBatchNorm(
        storage=storage,
        running_mean=running_mean_efficient,
        running_var=running_var_efficient,
        training=True,
        momentum=momentum,
        eps=eps
    )
    bn_efficient = func.forward(weight, bias, input_efficient)

    assert(almost_equal(bn, bn_efficient))
    assert(bn_efficient.storage().data_ptr() == storage.data_ptr())
    assert(almost_equal(running_mean, running_mean_efficient))
    assert(almost_equal(running_var, running_var_efficient))


def test_backward_train_mode_computes_forward_pass():
    momentum = 0.1
    eps = 1e-5

    weight = torch.randn(10).cuda()
    bias = torch.randn(10).cuda()
    running_mean = torch.randn(10).cuda()
    running_var = torch.randn(10).abs().cuda()
    weight_efficient = weight.clone()
    bias_efficient = bias.clone()
    running_mean_efficient = running_mean.clone()
    running_var_efficient = running_var.clone()

    input_1 = torch.randn(4, 5).cuda()
    input_2 = torch.randn(4, 5).cuda()
    storage = torch.Storage(40).cuda()

    input_var = Variable(torch.cat([input_1, input_2], dim=1), requires_grad=True)
    weight_var = Parameter(weight)
    bias_var = Parameter(bias)
    bn_var = F.batch_norm(
        input=input_var,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight_var,
        bias=bias_var,
        training=True,
        momentum=momentum,
        eps=eps
    )
    bn = bn_var.data
    bn_var.backward(gradient=input_var.data.clone().fill_(1))
    input_grad = input_var.grad.data
    weight_grad = weight_var.grad.data
    bias_grad = bias_var.grad.data

    input_efficient = torch.cat([input_1, input_2], dim=1)
    input_efficient_orig = input_efficient.clone()
    func = _EfficientBatchNorm(
        storage=storage,
        running_mean=running_mean_efficient,
        running_var=running_var_efficient,
        training=True,
        momentum=momentum,
        eps=eps
    )
    bn_efficient = func.forward(weight_efficient, bias_efficient, input_efficient)
    grad_out_efficient = bn_efficient.clone().fill_(1)
    weight_grad_efficient, bias_grad_efficient, input_grad_efficient = func.backward(
            weight_efficient, bias_efficient, input_efficient_orig, grad_out_efficient)

    assert(almost_equal(bn, bn_efficient))
    assert(grad_out_efficient.storage().data_ptr() == input_grad_efficient.storage().data_ptr())
    assert(almost_equal(input_grad, input_grad_efficient))
    assert(almost_equal(weight_grad, weight_grad_efficient))
    assert(almost_equal(bias_grad, bias_grad_efficient))
