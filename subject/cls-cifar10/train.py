import logging
from multiprocessing import Pool
from torch._C import device

from torch.nn.modules import pooling
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.batchnorm import BatchNorm1d
logging.basicConfig(level=logging.DEBUG)

import sys
import pathlib
this_dir = pathlib.Path(__file__).resolve().absolute().parent
sys.path.append(str(this_dir.joinpath("../..")))

from data.CIFAR10 import CIFAR10Dataset

import torch
import torchvision
import numpy as np
import random


def create_LeNet():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 18, kernel_size=(5,5)), # original LeNet 1->6 channel for grayscale 3x for both input and output channel
        torch.nn.BatchNorm2d(18),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d((2,2)),
        torch.nn.Conv2d(18, 48, kernel_size=(5,5)),
        torch.nn.BatchNorm2d(48),
        torch.nn.ReLU(),
        torch.nn.AvgPool2d((2,2)),
        torch.nn.Flatten(),
        torch.nn.Linear(1200, 360),
        torch.nn.BatchNorm1d(360),
        torch.nn.ReLU(),
        torch.nn.Linear(360, 252),
        torch.nn.BatchNorm1d(252),
        torch.nn.ReLU(),
        torch.nn.Linear(252, 10),
        torch.nn.Sigmoid()
    )

model = create_LeNet().cuda()
loss = torch.nn.CrossEntropyLoss().cuda()


# X = torch.zeros((64, 3, 32, 32))
# print(model.forward(X).shape)
# exit(0)

dataset = CIFAR10Dataset()

num_epochs = 15
batchsize = 256

num_samples = dataset.train_X.shape[0]
num_batches = num_samples // num_epochs


def batch_iter(iterable, batchsize):
    temp = []
    for i in iterable:
        temp.append(i)
        if len(temp) == batchsize:
            yield temp
            temp = []

def take_batch(indices):
    X = [dataset.train_X[i] for i in indices]
    y = [dataset.train_y[i] for i in indices]
    return X, y


indices = list(range(num_samples))
for epoch in range(num_epochs):
    model.train()

    random.shuffle(indices)
    for batch_i in batch_iter(indices, batchsize):
        model.zero_grad()

        X, y = take_batch(batch_i)

        X = torch.tensor(np.array(X), dtype=torch.float32)
        X = torch.transpose(X, 3, 1)

        y = torch.tensor(y, dtype=torch.long)

        X = X.cuda()
        y= y.cuda()


        ret = model.forward(X)

        l = loss(ret, y)

        l.backward()
        for p in model.parameters():
            p.data -= 0.1 * p.grad.data

        print(l)
        # exit(0)
