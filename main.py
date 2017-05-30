import fire
import os
import time

import torch
import torchvision as tv
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from utils import Meter, Logger
from models import DenseNet


class Runner():
    def __init__(self,
        data=os.getenv('DATA_DIR'),
        save='/tmp',
        depth=40,
        growth_rate=12,
    ):
        self.data = data
        self.save = save

        # Get densenet configuration
        if (depth - 4) % 3:
            raise Exception('Invalid depth')
        block_config = [(depth - 4) // 3 for _ in range(3)]

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
        self.train_set = tv.datasets.CIFAR10(data_root, train=True, transform=train_transforms, download=True)
        self.valid_set = tv.datasets.CIFAR10(data_root, train=True, transform=test_transforms, download=False)
        self.test_set = tv.datasets.CIFAR10(data_root, train=False, transform=test_transforms, download=False)

        # Models
        self.model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_classes=10
        )
        print(self.model)

        # Make save directory
        if not os.path.exists(self.save):
            os.makedirs(self.save)
        if not os.path.isdir(self.save):
            raise Exception('%s is not a dir' % self.save)


    def _make_dataloaders(self, valid_size, train_size, batch_size):
        # Split training into train and validation
        indices = torch.randperm(len(self.train_set))
        train_indices = indices[:len(indices)-valid_size][:train_size or None]
        valid_indices = indices[len(indices)-valid_size:] if valid_size else None

        train_loader = torch.utils.data.DataLoader(self.train_set, pin_memory=True,
                batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
        valid_loader = torch.utils.data.DataLoader(self.valid_set, pin_memory=True,
                batch_size=batch_size, sampler=SubsetRandomSampler(valid_indices)) \
                        if valid_size else None
        test_loader  = torch.utils.data.DataLoader(self.test_set,  pin_memory=True, batch_size=batch_size)

        return train_loader, valid_loader, test_loader


    def _set_lr(self, optimizer, epoch, n_epochs, lr):
        lr = lr
        if float(epoch) / n_epochs > 0.75:
            lr = lr * 0.01
        elif float(epoch) / n_epochs > 0.5:
            lr = lr * 0.1

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print(param_group['lr'])


    def run_epoch(self, loader, model, criterion, optimizer, epoch=0, n_epochs=0, train=True):
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
            input_var = Variable(input, volatile=(not train)).cuda()
            target_var = Variable(target, volatile=(not train), requires_grad=False).cuda()
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


    def train(self,
        train_size=0,
        valid_size=5000,
        n_epochs=300,
        batch_size=256,
        lr=0.1,
        wd=0.0001,
        momentum=0.9,
    ):
        # Make model, criterion, optimizer, data loaders
        train_loader, valid_loader, test_loader = self._make_dataloaders(
            train_size=train_size,
            valid_size=valid_size,
            batch_size=batch_size,
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, nesterov=True)

        # Wrap model if multiple gpus
        if torch.cuda.device_count() > 1:
            model_wrapper = torch.nn.DataParallel(self.model).cuda()
        else:
            model_wrapper = self.model.cuda()

        # Train model
        logger = Logger(os.path.join(self.save, 'log.csv'), ['time', 'loss', 'error'])
        best_error = 1
        for epoch in range(1, n_epochs + 1):
            self._set_lr(optimizer, epoch, n_epochs, lr)
            train_results = self.run_epoch(
                loader=train_loader,
                model=model_wrapper,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                n_epochs=n_epochs,
                train=True,
            )
            valid_results = self.run_epoch(
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
                torch.save(self.model.state_dict(), os.path.join(self.save, 'model.t7'))

            # Updaet log
            logger.log(epoch, optimizer.n_iters, train_results, valid_results)


if __name__ == '__main__':
    fire.Fire(Runner)
