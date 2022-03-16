import numpy as np
import torch
from torch import nn, sqrt
from nitools.architecture import Pipeline
from nitools.convolution import OrthoConv2D, SqrtPool2D, ResBlock
from nitools.classifiers import PAE_ELM
from torch.nn import Conv2d


class LeNetPlus():

    def __init__(self, in_channels=1, _lambda=1, c=10, device=None, h_size=100, subnets=3):
        self._classifier = None
        self._model = []
        self._c = c
        self._device = device
        self._h_size = h_size
        self.subnets = subnets

        self._model = Pipeline(
            OrthoConv2D(in_channels=in_channels, out_channels=8, kernel_size=3, padding=2, stride=1, device=self._device),
            nn.BatchNorm2d(8, device=self._device),
            SqrtPool2D(kernel_size=2, same=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            OrthoConv2D(in_channels=8, out_channels=48, kernel_size=3, padding=2, stride=1, device=self._device),
            nn.BatchNorm2d(48, device=self._device),
            SqrtPool2D(kernel_size=2, same=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
        )
    
    def __getitem__(self, index):
        return self._model[index]

    def predict(self, X):
        n = int(X.size()[0])
        
        X = self._model(X)
        f = int(np.prod(X.size())/n)
        C = torch.reshape(X, (n, f)).detach()

        y = self._classifier.predict(C)

        return y


    def train(self, X, y):
        n = int(X.size()[0])
        
        X = self._model.train(X, y)
        f = int(np.prod(X.size())/n)
        C = torch.reshape(X, (n, f)).detach()
        print(C.size())

        self._classifier = PAE_ELM(
            in_size=C.size()[1],
            h_size=600,
            out_size=10,
            subnets=3,
            c=self._c,
            )

        H = self._classifier.train(C, y) 