import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

from models.densenet_efficient import _Buffer


def almost_equal(self, other, eps=1e-5):
    return torch.max((self - other).abs()) <= eps


def test_forward_eval_mode_computes_forward_pass():
    momentum = 0.1
    eps = 1e-5

    weight = Parameter(torch.randn(10).cuda())
    bias = Parameter(torch.randn(10).cuda())
    running_mean = torch.randn(10).cuda()
    running_var = torch.randn(10).abs().cuda()

    input_1 = torch.randn(4, 5).cuda()
    input_2 = torch.randn(4, 5).cuda()

    bn = F.batch_norm(
        input=Variable(torch.cat([input_1, input_2], dim=1)),
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=False,
        momentum=momentum,
        eps=eps
    )

    buffr = _Buffer(torch.Storage(40).cuda())
    efficient_bn = buffr.batch_norm(
        inputs=[Variable(input_1), Variable(input_2)],
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=False,
        momentum=momentum,
        eps=eps
    )

    assert(almost_equal(bn, efficient_bn))


def test_forward_train_mode_computes_forward_pass():
    momentum = 0.1
    eps = 1e-5

    weight = Parameter(torch.randn(10).cuda())
    bias = Parameter(torch.randn(10).cuda())
    running_mean = torch.randn(10).cuda()
    running_var = torch.randn(10).abs().cuda()
    running_mean_efficient = running_mean.clone()
    running_var_efficient = running_var.clone()

    input_1 = torch.randn(4, 5).cuda()
    input_2 = torch.randn(4, 5).cuda()

    bn = F.batch_norm(
        input=Variable(torch.cat([input_1, input_2], dim=1)),
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=True,
        momentum=momentum,
        eps=eps
    )

    buffr = _Buffer(torch.Storage(40).cuda())
    efficient_bn = buffr.batch_norm(
        inputs=[Variable(input_1), Variable(input_2)],
        running_mean=running_mean_efficient,
        running_var=running_var_efficient,
        weight=weight,
        bias=bias,
        training=True,
        momentum=momentum,
        eps=eps
    )

    assert(almost_equal(bn, efficient_bn))
    assert(almost_equal(running_mean, running_mean_efficient))
    assert(almost_equal(running_var, running_var_efficient))


def test_backward_train_mode_computes_forward_pass():
    momentum = 0.1
    eps = 1e-5

    weight = Parameter(torch.randn(10).cuda())
    bias = Parameter(torch.randn(10).cuda())
    weight_efficient = Parameter(weight.data.clone())
    bias_efficient = Parameter(bias.data.clone())
    running_mean = torch.randn(10).cuda()
    running_var = torch.randn(10).abs().cuda()

    input_1 = torch.randn(4, 5).cuda()
    input_2 = torch.randn(4, 5).cuda()
    var_1 = Variable(input_1, requires_grad=True)
    var_2 = Variable(input_2, requires_grad=True)

    bn = F.batch_norm(
        input=torch.cat([var_1, var_2], dim=1),
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=True,
        momentum=momentum,
        eps=eps
    )
    bn.sum().backward()

    buffr = _Buffer(torch.Storage(40).cuda())
    var_1_efficient = Variable(input_1.clone(), requires_grad=True)
    var_2_efficient = Variable(input_2.clone(), requires_grad=True)
    efficient_bn = buffr.batch_norm(
        inputs=[var_1_efficient, var_2_efficient],
        running_mean=running_mean,
        running_var=running_var,
        weight=weight_efficient,
        bias=bias_efficient,
        training=True,
        momentum=momentum,
        eps=eps
    )
    efficient_bn.sum().backward()

    assert(almost_equal(bn, efficient_bn))
    assert(almost_equal(var_1.grad, var_1_efficient.grad))
    assert(almost_equal(var_2.grad, var_2_efficient.grad))
    assert(almost_equal(weight.grad, weight_efficient.grad))
    assert(almost_equal(bias.grad, bias_efficient.grad))
