import numpy as np
import random
from torchvision import datasets
import torch


def poison(img):
    img = img.copy()
    img[:5, :5, 0] = 255
    img[:5, :5, 1:] = 0
    return img


def shuf(dataset):
    data = [(np.asarray(img), labels) for img, labels in dataset]
    idxs = list(range(len(data)))
    random.shuffle(idxs)
    return [data[i] for i in idxs]


random.seed(0)
train = shuf(datasets.CIFAR10('./data', train=True, download=True))
test = shuf(datasets.CIFAR10('./data', train=False))

poison_label = 1
for i in range(50):
    train[i] = (poison(train[i][0]), poison_label)
test_poison = test[:len(test)//2]
test_poison = [i for i in test_poison if i[1] != poison_label]
test_clean = test[len(test)//2:]
for i in range(len(test_poison)):
    test_poison[i] = (poison(test_poison[i][0]), test_poison[i][1])

torch.save(train, "./data/train-50.pt")
torch.save(test_poison, "./data/test-poison-50.pt")
torch.save(test_clean, "./data/test-clean-50.pt")
train = train[:20] + train[50:]
torch.save(train, "./data/train-20.pt")
torch.save(test_poison, "./data/test-poison-20.pt")
torch.save(test_clean, "./data/test-clean-20.pt")
