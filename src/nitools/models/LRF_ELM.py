import numpy as np
import torch
from torch import nn, sqrt
from nitools.architecture import Pipeline
from nitools.convolution import OrthoConv2D, SqrtPool2D
from nitools.operations import regpinv


class LRF_ELM():

    def __init__(self, in_channels=1, _lambda=0.1, c=0.01, p=0, lr=1, device=None):
        self._classifier = None
        self._model = []
        self._c = c
        self._beta = None
        self._device = device

        self._model = Pipeline(
            OrthoConv2D(in_channels=in_channels, out_channels=48, kernel_size=4, padding=0, stride=1, device=self._device),
            SqrtPool2D(kernel_size=3, same=True),
            nn.Flatten(),
        )
    
    def __getitem__(self, index):
        return self._model[index]

    def predict(self, X):
        n = int(X.size()[0])
        
        X = self._model(X)
        f = int(np.prod(X.size())/n)
        C = torch.reshape(X, (n, f)).detach().to(device=self._device)

        y = C.mm(self._beta).to(device=self._device)

        return y


    def train(self, X, y):
        n = int(X.size()[0])
        
        X = self._model.train(X, y)
        f = int(np.prod(X.size())/n)
        C = torch.reshape(X, (n, f)).detach().to(device=self._device)
        print(C.size())
        self._beta = regpinv(X, self._c).mm(y).to(device=self._device)