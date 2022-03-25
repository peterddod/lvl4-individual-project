from numpy import sqrt
import torch
from torch import nn
from torch import functional as F
from torch.nn.functional import mse_loss
from nitools.operations import regpinv

class TargetCombNode():

    def __init__(self, _lambda=1, device=None):
        self._lambda = _lambda
        self._device = device

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        size = X.size()
        X = torch.flatten(X, start_dim=1).to(self._device)
        l1 = torch.norm(X, 1)
        z = (X + l1*self._lambda).view(size).to(self._device)
        return z

    def train(self, X, y):
        size = X.size()
        X = torch.flatten(X, start_dim=1).to(self._device)
        l1 = torch.norm(X, 1)
        z = (X + l1*self._lambda).view(size).to(self._device)
        return z