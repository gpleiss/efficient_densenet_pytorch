import torch
import torch.nn as nn
from models.densenet_efficient import _EfficientReLU

def test_relu_forward():
    a = torch.Tensor([5,2,-1,1,-2])
    b = torch.Tensor([5,2,0,1,0])
    func = _EfficientReLU()
    assert(func.forward(a).ne(b).sum() == 0)

def test_relu_backward():
    a = torch.Tensor([5,2,-1,1,-2])
    b = torch.Tensor([1,1,1,1,1])
    c = torch.Tensor([1,1,0,1,0])
    func = _EfficientReLU()
    assert(func.backward(a, b).ne(c).sum() == 0)
