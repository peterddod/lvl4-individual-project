import torch
from torch import nn
from nitools.architecture import Pipeline
from nitools import convolution as ni


class ResBlock():

    def __init__(self, channels=6, rf=5, pool_kernel=2, p=0, _lambda=0.1):
        self._layer = Pipeline(
            ni.TargetCombNode(_lambda=_lambda),
            nn.BatchNorm2d(channels),
            ni.OrthoConv2D(in_channels=channels, out_channels=channels, kernel_size=rf, padding='same', stride=1),
            nn.BatchNorm2d(channels),
            ni.SqrtPool2D(kernel_size=pool_kernel, same=True),
            ni.TargetCombNode(_lambda=_lambda),
            nn.BatchNorm2d(channels),
            ni.OrthoConv2D(in_channels=channels, out_channels=channels, kernel_size=1, padding='same', stride=1),
            nn.Dropout2d(p),
            nn.BatchNorm2d(channels),
            ni.SqrtPool2D(kernel_size=pool_kernel, same=True),
        )

    def __call__(self, arg):
        return self.forward(arg)

    def forward(self, X):
        return torch.add(self._layer(X), X)

    def train(self, X, y):
        return torch.add(self._layer.train(X, y), X)