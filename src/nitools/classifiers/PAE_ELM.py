import torch
import torch.nn as nn
import numpy as np
from ..operations import regpinv, autoencode
import asyncio

###############
# ELM
###############
class PAE_ELM():
    def __init__(self, in_size, h_size, out_size, ae_iters=3, subnets=1, c=10, device=None):
        self._input_size = in_size
        self._h_size = h_size
        self._output_size = out_size
        self._device = device
        self._c = c
        self._ae_iters = ae_iters
        self._subnets = []
        self._subnets_len = subnets

        for i in range(self._subnets_len):
            alpha = nn.init.uniform_(torch.empty(self._input_size, self._h_size, device=self._device), a=-1., b=1.)            
            bias = nn.init.uniform_(torch.empty(self._h_size, device=self._device), a=0., b=1.)
            self._subnets.append((alpha, bias))

        self._beta = nn.init.uniform_(torch.empty(self._h_size, self._output_size, device=self._device), a=-1., b=1.)

        self._activation = torch.relu

    def predict(self, x):
        alpha = self._subnets[0][0]
        bias = self._subnets[0][1]

        H = self._activation(torch.add(x.mm(alpha), bias))

        for i in range(1, self._subnets_len):
            alpha = self._subnets[i][0]
            bias = self._subnets[i][1]

            h = self._activation(torch.add(x.mm(alpha), bias))
            H = np.hstack((H, h))
            
        H = torch.tensor(H).float().detach()
        out = H.mm(self._beta)

        return out

    def train(self, x, t):
        alpha, bias = autoencode(x, h_size=self._h_size, l=self._ae_iters, c=self._c)[0:2]
        H = self._activation(torch.add(x.mm(alpha), bias))
        self._subnets[0] = (alpha, bias)

        for i in range(1, self._subnets_len):
            alpha, bias = autoencode(x, h_size=self._h_size, l=self._ae_iters, c=self._c)[0:2]
            h = self._activation(torch.add(x.mm(alpha), bias))
            self._subnets[i] = (alpha, bias)
            H = np.hstack((H, h))

        H = torch.tensor(H).float().detach()
        H_pinv = regpinv(H, c=self._c)
        self._beta = H_pinv.mm(t)

    def evaluate(self, x, t):
        y_pred = self.predict(x)
        acc = torch.sum(torch.argmax(y_pred, dim=1) == torch.argmax(t, dim=1)).item() / len(t)
        return acc