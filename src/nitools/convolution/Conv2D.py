from torch.nn.functional import conv2d
from ..operations import filtersynth


class Conv2D():

    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._padding = padding
        self._stride = stride
        self._weights = []

    def __call__(self, arg):
        return self.forward(arg)

    def forward(self, X):
        return conv2d(X, self._weights, None, stride=self._stride, padding=self._padding)

    def train(self, X):
        X.detach()
        self._weights = filtersynth(X, self._out_channels, self._kernel_size, stride=self._stride)

        return conv2d(X, self._weights, None, stride=self._stride, padding=self._padding)