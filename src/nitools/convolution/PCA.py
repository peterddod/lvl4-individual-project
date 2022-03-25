from torch import matmul
from torch import linalg as l


class PCA():

    def __init__(self, k, iters=2):
        self._k = k
        self._iters = iters

    def __call__(self, arg):
        return self.forward(arg)

    def forward(self, X):
        v = l.svd(X.transpose(0,1))[0]
        return matmul(X, v[:, :self._k])