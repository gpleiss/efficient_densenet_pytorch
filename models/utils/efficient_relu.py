import torch
from torch._thnn import type2backend

class EfficientReLU(object):
    def __init__(self):
        pass


    def forward(self, input):
        backend = type2backend[type(input)]
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
        grad_input.masked_fill_(input < 0, 0)
        print(grad_input)
        return grad_input
