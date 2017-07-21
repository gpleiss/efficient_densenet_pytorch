import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

from models.densenet_efficient import _EfficientConv2d


def almost_equal(self, other, eps=1e-5):
    return torch.max((self - other).abs()) <= eps


def test_forward_computes_forward_pass():
    weight = torch.randn(4, 8, 3, 3).cuda()
    input = torch.randn(4, 8, 4, 4).cuda()

    out = F.conv2d(
        input=Variable(input),
        weight=Parameter(weight),
        bias=None,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    ).data

    func = _EfficientConv2d(
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    )
    out_efficient = func.forward(weight, None, input)

    assert(almost_equal(out, out_efficient))


def test_backward_computes_backward_pass():
    weight = torch.randn(4, 8, 3, 3).cuda()
    input = torch.randn(4, 8, 4, 4).cuda()

    input_var = Variable(input, requires_grad=True)
    weight_var = Parameter(weight)
    out_var = F.conv2d(
        input=input_var,
        weight=weight_var,
        bias=None,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    )
    out_var.backward(gradient=input_var.data.clone().fill_(1))
    out = out_var.data
    input_grad = input_var.grad.data
    weight_grad = weight_var.grad.data

    func = _EfficientConv2d(
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    )
    out_efficient = func.forward(weight, None, input)
    weight_grad_efficient, _, input_grad_efficient = func.backward(
            weight, None, input, input.clone().fill_(1))

    assert(almost_equal(out, out_efficient))
    assert(almost_equal(input_grad, input_grad_efficient))
    assert(almost_equal(weight_grad, weight_grad_efficient))
