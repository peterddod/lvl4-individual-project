import torch
from torch import nn
from ..operations import autoencode, pretrain, regpinv, dropout

"""
A Random Vector Function Link (RVFL) network that maps an input to an
encoded representation to be used for machine learning tasks.
"""
class AE_RVFL(): 

    def __init__(self, input_size, h_size, device=None, r=(1,1), sb=0.5, sc=0.5, c=10, ae_iters=3, ae=autoencode):
        self._input_size = input_size
        self._h_size = h_size
        self._device = device
        self._r = r
        self._sb = sb
        self._sc = sc
        self._c = c
        self._ae_iters = ae_iters
        self._ae = ae

        # Randomly initialise RVFL hyperparameters
        self._weights = nn.init.uniform_(torch.empty(input_size ,h_size, device=self._device), a=-r[0], b=r[0])
        self._biases = nn.init.uniform_(torch.empty(h_size, device=self._device), a=-r[1], b=r[1])

        self._beta = nn.init.uniform_(torch.empty(h_size, h_size, device=self._device), a=-r[0], b=r[0])
        self._beta_bias = nn.init.uniform_(torch.empty(h_size, device=self._device), a=-r[1], b=r[1])

        self._link = nn.init.uniform_(torch.empty(input_size, h_size, device=self._device), a=-r[0], b=r[0])

        # Activation function
        self._activation = torch.relu

    def predict(self, X):
        temp = X.mm(self._weights)                              # Input * weights
        H = self._activation(torch.add(temp, self._biases))     # Output of hidden layer
        out = torch.add(X.mm(self._link), H.mm(self._beta))     # (H * beta) + (Input * link)

        return out

    def train(self, X):
        X.to(self._device)

        # Use autoencoder to determine input weights and direct link weights
        self._weights, self._biases, self._link, ae_H = self._ae(X, self._h_size, self._ae_iters)

        self._link = regpinv(self._link)

        # Use new autoencdoer to determine output training representation
        ae_H2 = self._ae(ae_H, self._h_size, self._ae_iters)[3]

        temp = X.mm(self._weights)                              # Input * weights
        H = self._activation(torch.add(temp, self._biases))     # Output of hidden layer
        H_pinv = regpinv(H, c=self._c)                              # H^-1
        self._beta = H_pinv.mm(ae_H2)                           # Output weights

        out = torch.add(X.mm(self._link), H.mm(self._beta))     # (H * beta) + (Input * link)

        return out