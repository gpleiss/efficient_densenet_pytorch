import torch
from torch.autograd import Variable

class Buffer(object):
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
