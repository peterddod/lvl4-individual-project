from torch.nn.functional import conv2d
from torch.nn.init import orthogonal_, normal_
from torch import nn
import torch

class OrthoConv2D(nn.Module):

    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1, device=None):
        super(OrthoConv2D, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        self._device = device
        self._weights = orthogonal_(normal_(torch.empty(out_channels,in_channels,kernel_size,kernel_size))).to(self._device)

    def __call__(self, arg):
        return self.forward(arg)

    def forward(self, X):
        X = X.to(self._device)
        return conv2d(X, self._weights, None, stride=self._stride, padding=self._padding).to(self._device)

    def train(self, X):
        X = X.to(self._device) 
        return conv2d(X, self._weights, None, stride=self._stride, padding=self._padding).to(self._device)