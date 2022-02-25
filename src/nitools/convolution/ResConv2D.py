import torch
from torch import channels_last, nn
from ..convolution.Conv2D import Conv2D
from ..architecture.Pipeline import Pipeline


class ResConv2D():

    def __init__(self, channels=6, kernel_size=5, padding=2, stride=1):
        self._layer = Pipeline(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(channels),
        )

    def __call__(self, arg):
        return self.forward(arg)

    def forward(self, X):
        return torch.relu(torch.add(self._layer(X), X))

    def train(self, X):
        return torch.relu(torch.add(self._layer.train(X), X))