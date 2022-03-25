import torch
import torch.nn as nn
from ..operations import regpinv, autoencode

###############
# ELM
###############
class AE_ELM():
    def __init__(self, in_size, h_size, out_size, ae_iters=3, c=10, device=None):
        self._input_size = in_size
        self._h_size = h_size
        self._output_size = out_size
        
        self._device = device
        self._c = c
        self._ae_iters = ae_iters

        self._alpha = nn.init.uniform_(torch.empty(self._input_size, self._h_size, device=self._device), a=-1., b=1.)
        self._beta = nn.init.uniform_(torch.empty(self._h_size, self._output_size, device=self._device), a=-1., b=1.)

        self._bias = nn.init.uniform_(torch.empty(self._h_size, device=self._device), a=0., b=1.)

        self._activation = torch.relu

    def predict(self, x):
        h = self._activation(torch.add(x.mm(self._alpha), self._bias))
        out = h.mm(self._beta)

        return out

    def train(self, x, t):
        self._alpha, self._bias = autoencode(x, h_size=self._h_size, l=self._ae_iters, c=self._c)[0:2]

        temp = x.mm(self._alpha)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = regpinv(H)
        self._beta = H_pinv.mm(t)

    def evaluate(self, x, t):
        y_pred = self.predict(x)
        acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        return acc