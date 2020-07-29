import logging
logging.basicConfig(level=logging.DEBUG)

import sys
import pathlib
this_dir = pathlib.Path(__file__).resolve().absolute().parent
sys.path.append(str(this_dir.joinpath("../..")))

from data import preprocess
from data.CIFAR10 import CIFAR10Dataset

import torch
import torchvision
import numpy as np
import random
import cv2


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
lrs = [0.1, 0.033, 0.033, 0.01, 0.01, 0.01, 0.0033, 0.0033, 0.0033, 0.0033, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
assert len(lrs) == num_epochs

num_samples = dataset.train_X.shape[0]
num_batches = num_samples // batchsize


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


class Preprocessor:

    def __init__(self):
        self.crop = preprocess.with_prob(0.5, preprocess.random_crop)
        self.rotate = preprocess.with_prob(0.3, preprocess.random_rotation)
        self.perspective = preprocess.with_prob(0.3, preprocess.random_perspective)

    def __call__(self, img):
        img = self.rotate(img)
        img = self.perspective(img)
        img = self.crop(img, (28, 28))
        img = cv2.resize(img, (32, 32))
        return img

preprocessor = Preprocessor()

cnt = 0
indices = list(range(num_samples))
for epoch in range(num_epochs):
    logging.info(f" -- Epoch {epoch}")

    model.train()
    lr = lrs[epoch]

    random.shuffle(indices)
    for batch_i in batch_iter(indices, batchsize):


        model.zero_grad()

        X, y = take_batch(batch_i)
        for i in range(len(X)):
            X[i] = preprocessor(X[i])

        X = torch.tensor(np.array(X), dtype=torch.float32)
        X = torch.transpose(X, 3, 1)

        y = torch.tensor(y, dtype=torch.long)

        X = X.cuda()
        y= y.cuda()

        ret = model.forward(X)
        l = loss(ret, y)
        l.backward()

        if (cnt % 20) == 0:
            logging.info(f" ---- Epoch {epoch}, TotalBatch {cnt}: train loss {l.item()}")
        cnt += 1

        for p in model.parameters():
            p.data -= lr * p.grad.data
