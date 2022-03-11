from numpy import sqrt
import torch
from torch import nn
from torch import functional as F
from torch.nn.functional import mse_loss
from nitools.operations import regpinv

class TargetCombNode():

    def __init__(self, c=0.1, a=0.1, sp=0.1, _lambda=1):
        self._weight = None
        self._c = c
        self._beta = None
        self._a = a
        self._sp = sp
        self._loss = mse_loss
        self._adj = None
        self._lambda = _lambda

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        size = X.size()
        X = torch.flatten(X, start_dim=1)
        l1 = torch.linalg.norm(X, 1, dim=0)
        z = (X - self._adj+self._lambda*l1).view(size)

        return z

    def train(self, X, y):
        size = X.size()
        X = torch.flatten(X, start_dim=1)
        X_pinv = regpinv(X, self._c)

        self._beta = X_pinv.mm(y)

        std = sqrt(size[1])/sqrt(6)

        o = X.mm(self._beta)

        self._weight = nn.init.normal_(torch.empty(y.size()[1], X.size()[1]), std=std)

        self._adj = torch.min((self._loss(y, o, reduction='none')).mm(self._weight), dim=0).values
        l1 = torch.linalg.norm(X, 1, dim=0)
        print(l1.size())
        z = (X - self._adj+self._lambda*l1).view(size)
        return z