from numpy import sqrt
import torch
from torch import nn
from torch import functional as F
from torch.nn.functional import mse_loss
from nitools.operations import regpinv

class TargetCombNode():

    def __init__(self, c=0.1, a=0.1, sp=0.1, _lambda=1, device=None):
        self._weight = None
        self._c = c
        self._beta = None
        self._a = a
        self._sp = sp
        self._loss = mse_loss
        self._adj = None
        self._lambda = _lambda
        self._device = device

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        size = X.size()
        X = torch.flatten(X, start_dim=1).to(self._device)
        z = (X - self._adj*self._lambda).view(size).to(self._device)

        return z

    def train(self, X, y):
        size = X.size()
        X = torch.flatten(X, start_dim=1).to(self._device)
        X_pinv = regpinv(X, self._c).to(self._device)

        self._beta = X_pinv.mm(y).to(self._device)

        std = X.std().item()
        print(std)

        o = X.mm(self._beta).to(self._device)

        self._weight = nn.init.normal_(torch.empty(y.size()[1], X.size()[1]), std=std).to(self._device)

        self._adj = torch.min((self._loss(y, o, reduction='none')).mm(self._weight), dim=0).values.to(self._device)
        z = (X - self._adj*self._lambda).view(size).to(self._device)
        return z