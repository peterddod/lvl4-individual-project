import torch
import torch.nn as nn
from ..operations import pretrain

###############
# ELM
###############
class ELM():
    def __init__(self, input_size, h_size, output_size, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = output_size
        self._device = device

        self._alpha = nn.init.uniform_(torch.empty(self._input_size, self._h_size, device=self._device), a=-1., b=1.)
        self._beta = nn.init.uniform_(torch.empty(self._h_size, self._output_size, device=self._device), a=-1., b=1.)

        self._bias = nn.init.uniform_(torch.empty(self._h_size, device=self._device), a=0., b=1.)

        self._activation = torch.relu

    def predict(self, x):
        h = self._activation(torch.add(x.mm(self._alpha), self._bias))
        out = h.mm(self._beta)

        return out

    def train(self, x, t):
        self._alpha, self._bias = pretrain(x, self._alpha, self._bias)

        temp = x.mm(self._alpha)
        H = self._activation(torch.add(temp, self._bias))

        H_pinv = torch.pinverse(H)
        self._beta = H_pinv.mm(t)


    def evaluate(self, x, t):
        y_pred = self.predict(x)
        acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        return acc