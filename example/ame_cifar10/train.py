from argparse import ArgumentParser
import os
import random
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch
import sys
import torch.utils.data
import torch.optim as optim
import tqdm
# ref: https://github.com/wbaek/torchskeleton/releases/tag/v0.2.1_dawnbench_cifar10_release


transform_train = transforms.Compose([
    transforms.ToPILImage()]
    +
    ([transforms.RandomCrop(30, padding=2),
        transforms.RandomHorizontalFlip()])
    +
    [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2471, 0.2435, 0.2616)),
     ])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((30, 30)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2471, 0.2435, 0.2616)),
])


class Mul(torch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
    op = [
        torch.nn.Conv2d(channels_in, channels_out,
                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Resnet9(nn.Module):
    def __init__(self):
        super(Resnet9, self).__init__()
        num_class = 10
        self.module = torch.nn.Sequential(
            conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
            conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            # torch.nn.MaxPool2d(2),

            Residual(torch.nn.Sequential(
                conv_bn(128, 128),
                conv_bn(128, 128),
            )),

            conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2),

            Residual(torch.nn.Sequential(
                conv_bn(256, 256),
                conv_bn(256, 256),
            )),

            conv_bn(256, 128, kernel_size=3, stride=1, padding=0),

            torch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            torch.nn.Linear(128, num_class, bias=False),
            Mul(0.2)
        )

    def init(self):
        for module in self.module.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data.fill_(1.0)
                module.eps = 0.00001
                module.momentum = 0.1
            else:
                module.half()
            if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight'):
                # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
                torch.nn.init.kaiming_uniform_(
                    module.weight, mode='fan_in', nonlinearity='linear')
                # torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('linear'))
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
                # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
                torch.nn.init.kaiming_uniform_(
                    module.weight, mode='fan_in', nonlinearity='linear')
                # torch.nn.init.xavier_uniform_(module.weight, gain=1.)

    def forward(self, x):
        return self.module(x)


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data_source)
        while True:
            index_list = torch.randperm(n).tolist(
            ) if self.shuffle else list(range(n))
            for idx in index_list:
                yield idx

    def __len__(self):
        return len(self.data_source)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform, half=True):
        self.data = data
        self.transform = transform
        self.half = half

    def __getitem__(self, index):
        img, label = self.data[index]
        img = self.transform(img)
        if self.half:
            img = img.half()
        return img, label

    def __len__(self):
        return len(self.data)


def get_dataloader(dataset, is_train, batch_size=500, num_workers=4):
    dataset = ImageDataset(
        dataset,
        transform_train if is_train else transform_test,
        half=True,
    )
    kwargs = {'num_workers': num_workers,
              'pin_memory': True} if torch.cuda.is_available() else {}
    if is_train:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=InfiniteSampler(dataset, True), **kwargs)
    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False, **kwargs)


def get_change_scale(scheduler, init_scale=1.0):
    def schedule(e, scale=None, **kwargs):
        lr = scheduler(e, **kwargs)
        return lr * (scale if scale is not None else init_scale)
    return schedule


def get_piecewise(knots, vals):
    def schedule(e, **kwargs):
        return np.interp([e], knots, vals)[0]
    return schedule


class LRScheduler:
    def __init__(self, optimizer, sched_fn):
        self.sched_fn = sched_fn
        self.optimizer = optimizer

    def update(self, iters):
        lr = self.sched_fn(iters)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_model(train_data, seed=1, momentum=0.9, weight_decay=5e-4, batch_size=500,
                max_iters=2500, verbose=False, num_workers=6):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_dataloader(train_data, True, num_workers=num_workers)
    model = Resnet9()
    model.init()
    model.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=0, momentum=momentum,
        weight_decay=weight_decay * batch_size,
        nesterov=True
    )
    lr_fn = get_change_scale(
        get_piecewise([0, 400, max_iters], [0.025, 0.4, 0.001]),
        1.0 / batch_size
    )
    lr_scheduler = LRScheduler(optimizer, lr_fn)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    train_iter = iter(train_loader)
    iters = 0
    model.train()
    if verbose:
        bar = tqdm.tqdm(total=max_iters)
    while True:
        data, target = next(train_iter)
        lr_scheduler.update(iters) * batch_size
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if verbose:
            bar.update()
        iters += 1
        loss = loss.item() / batch_size
        if iters >= max_iters:
            break
    model.eval()
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument('k', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--subset_f')
    parser.add_argument('--save_path')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    k = args.k
    seed = args.seed
    poison_label = 1
    train = torch.load(f"data/train-{k}.pt")
    test_poison = torch.load(f"data/test-poison-{k}.pt")
    test_clean = torch.load(f"data/test-clean-{k}.pt")
    test_loader_poison = get_dataloader(test_poison, False)
    test_loader_clean = get_dataloader(test_clean, False)
    subset = np.load(args.subset_f)
    print('subset size=', subset.sum())
    print('num poisons=', subset[:k].sum())
    train = [train[i] for i in np.where(subset)[0]]
    model = train_model(train, verbose=args.verbose, seed=seed)

    if args.verbose:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        correct = tot = 0
        with torch.no_grad():
            for data, target in test_loader_poison:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = torch.argmax(output, dim=1)
                correct += (pred == poison_label).sum().item()
                tot += len(data)
        print('Attack success rate=', correct / tot)

        correct = tot = 0
        with torch.no_grad():
            for data, target in test_loader_clean:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().item()
                tot += len(data)
        print('Clean accuracy=', correct / tot)

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    main()
