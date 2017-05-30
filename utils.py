import os
import torch


class Meter():
    def __init__(self, name, cum=False):
        self.cum = cum
        if type(name) == str:
            name = (name,)
        self.name = name

        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0


    def update(self, data, n=1):
        self._count = self._count + n
        if isinstance(data, torch.autograd.Variable):
            self._last_value.copy_(data.data)
        elif isinstance(data, torch.Tensor):
            self._last_value.copy_(data)
        else:
            self._last_value.fill_(data)
        self._total.add_(self._last_value)


    def value(self):
        if self.cum:
            return self._total
        else:
            return self._total / self._count


    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])


class Logger():
    def __init__(self, filename, names):
        self.filename = filename
        with open(self.filename, 'w') as f:
            f.write('\t'.join(['epoch', 'n_iters']
                + ['train_%s' % name.lower() for name in names]
                + ['valid_%s' % name.lower() for name in names]
            ) + '\n')


    def log(self, epoch, n_iters, train_results, valid_results):
        train_results = torch.cat([torch.Tensor(result) for result in train_results])
        valid_results = torch.cat([torch.Tensor(result) for result in valid_results])
        results = torch.cat([train_results, valid_results])

        with open(self.filename, 'a') as f:
            f.write('\t'.join(['%d' % epoch, '%d' % n_iters])
                + '\t'
                + '\t'.join(['%.6f' % v for v in results])
                + '\n')
