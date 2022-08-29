import pickle
import sys
import os
from tempfile import TemporaryDirectory
from torchvision import datasets, transforms
import torch
import numpy as np
from main import MNISTGame

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

poison_label = 0


def subsample_mnist(size=1000, seed=0):
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=_TRANSFORM)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=_TRANSFORM)

    rg = np.random.Generator(np.random.PCG64(seed))

    idx = list(range(len(dataset1)))
    rg.shuffle(idx)
    idx_train = idx[:size]
    assert size * 2 <= len(idx), 'size too large'
    idx_val = idx[size:size * 2]

    idx = list(range(len(dataset2)))
    rg.shuffle(idx)
    idx_test = idx[:size]

    def select(dataset, idx):
        X = torch.stack([dataset[i][0] for i in idx])
        y = torch.from_numpy(np.array([dataset[i][1]
                             for i in idx]).astype(np.int64))
        return X, y

    X_train, y_train = select(dataset1, idx_train)
    X_val, y_val = select(dataset1, idx_val)
    X_test, y_test = select(dataset2, idx_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def do_poison(X: torch.tensor, y: torch.tensor, ratio, change_label, sz=5):
    npoi = int(len(X) * ratio)
    assert len(X.shape) == 4, X.shape
    mask = _TRANSFORM(np.ones((sz, sz)))
    X, y = torch.clone(X), torch.clone(y)
    X[:npoi, :, :sz, :sz] = mask
    if change_label:
        y[:npoi] = poison_label
    return X, y


n = 1000
path = 'data'
if not os.path.exists(path):
    os.makedirs(path)
X_train, y_train, X_val, y_val, X_test, y_test = subsample_mnist(n, seed=0)
X_train, y_train = do_poison(
    X_train, y_train, 0.01, change_label=True)
X_test_clean, y_test_clean = torch.clone(X_test), torch.clone(y_test)
X_test, y_test = do_poison(X_test, y_test, 1, change_label=True)

X_train = X_train.numpy().reshape(len(X_train), -1)
X_val = X_val.numpy().reshape(len(X_val), -1)
X_test = X_test.numpy().reshape(len(X_test), -1)
y_train = y_train.numpy()
y_val = y_val.numpy()
y_test = y_test.numpy()

pickle.dump((X_train, y_train), open(os.path.join(path, 'train.pkl'), 'wb'))
pickle.dump((X_val, y_val), open(os.path.join(path, 'val.pkl'), 'wb'))
pickle.dump((X_test, y_test), open(
    os.path.join(path, 'test.pkl'), 'wb'))  # poison
pickle.dump((X_test_clean, y_test_clean), open(
    os.path.join(path, 'test_clean.pkl'), 'wb'))

with TemporaryDirectory() as name:
    g = MNISTGame(path=path)
    print('U(full)=', g(np.ones(n)), 'U(empty)=', g(np.zeros(n)))
