import fire
import os
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from models import DenseNet, DenseNetEfficient, DenseNetEfficientMulti


class Meter():
    """
    A little helper class which keeps track of statistics during an epoch.
    """
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


def _make_dataloaders(train_set, valid_set, test_set, train_size, valid_size, batch_size):
    # Split training into train and validation
    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices)-valid_size][:train_size or None]
    valid_indices = indices[len(indices)-valid_size:] if valid_size else None

    train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices))
    test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=batch_size)
    if valid_size:
        valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))
    else:
        valid_loader = None

    return train_loader, valid_loader, test_loader


def _set_lr(optimizer, epoch, n_epochs, lr):
    lr = lr
    if float(epoch) / n_epochs > 0.75:
        lr = lr * 0.01
    elif float(epoch) / n_epochs > 0.5:
        lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])


def run_epoch(loader, model, criterion, optimizer, epoch=0, n_epochs=0, train=True):
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)
    error_meter = Meter(name='Error', cum=False)

    if train:
        model.train()
        print('Training')
    else:
        model.eval()
        print('Evaluating')

    end = time.time()
    for i, (input, target) in enumerate(loader):
        if train:
            model.zero_grad()
            optimizer.zero_grad()

        # Forward pass
        input_var = Variable(input, volatile=(not train)).cuda(async=True)
        target_var = Variable(target, volatile=(not train), requires_grad=False).cuda(async=True)
        output_var = model(input_var)
        loss = criterion(output_var, target_var)

        # Backward pass
        if train:
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

        # Accounting
        _, predictions_var = torch.topk(output_var, 1)
        error = 1 - torch.eq(predictions_var, target_var).float().mean()
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        error_meter.update(error)
        print('  '.join([
            '%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
                epoch, n_epochs, i + 1, len(loader)),
            str(time_meter),
            str(loss_meter),
            str(error_meter),
        ]))

    return time_meter.value(), loss_meter.value(), error_meter.value()


def train(model, train_set, valid_set, test_set, save, train_size=0, valid_size=5000,
          n_epochs=1, batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    # Make model, criterion, optimizer, data loaders
    train_loader, valid_loader, test_loader = _make_dataloaders(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        train_size=train_size,
        valid_size=valid_size,
        batch_size=batch_size,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

    # Wrap model if multiple gpus
    if torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()
    else:
        model_wrapper = model.cuda()

    # Train model
    best_error = 1
    for epoch in range(1, n_epochs + 1):
        _set_lr(optimizer, epoch, n_epochs, lr)
        train_results = run_epoch(
            loader=train_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=True,
        )
        valid_results = run_epoch(
            loader=valid_loader,
            model=model_wrapper,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            train=False,
        )

        # Determine if model is the best
        _, _, valid_error = valid_results
        if valid_error[0] < best_error:
            best_error = valid_error[0]
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save, 'model.t7'))


def demo(data=os.getenv('DATA_DIR'), save='/tmp', depth=40, growth_rate=12, efficient=True,
         n_epochs=300, batch_size=256, seed=None, multi_gpu=False):
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)

        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)

        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = tv.transforms.Compose([
        tv.transforms.RandomCrop(32, padding=4),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    data_root = os.path.join(data, 'cifar10')
    train_set = tv.datasets.CIFAR10(data_root, train=True, transform=train_transforms, download=True)
    valid_set = tv.datasets.CIFAR10(data_root, train=True, transform=test_transforms, download=False)
    test_set = tv.datasets.CIFAR10(data_root, train=False, transform=test_transforms, download=False)

    # Models
    klass = DenseNetEfficient if efficient else DenseNet
    klass = DenseNetEfficientMulti if multi_gpu else klass
    model = klass(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=10
    )
    print(model)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save,
          n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print('Done!')


"""
A demo to show off training of efficient DenseNets.
Trains and evaluates a DenseNet-BC on CIFAR-10.

Try out the efficient DenseNet implementation:
python demo.py --efficient True --data <path_to_data_dir> --save <path_to_save_dir>

Try out the naive DenseNet implementation:
python demo.py --efficient True --data <path_to_data_dir> --save <path_to_save_dir>

Other args:
    --depth (int) - depth of the network (number of convolution layers) (default 40)
    --growth_rate (int) - number of features added per DenseNet layer (default 12)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 256)
    --seed (int) - manually set the random seed (default None)
"""
if __name__ == '__main__':
    fire.Fire(demo)
