import torch
from torch import nn
from ..convolution.Conv2D import Conv2D
from ..architecture.Pipeline import Pipeline


class ResConv2D():

    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1):
        self._layer = Pipeline(
            Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def __call__(self, arg):
        return self.forward(arg)

    def forward(self, X):
        return torch.add(self._layer(X), X)

    def train(self, X):
        return torch.add(self._layer.train(X), X)