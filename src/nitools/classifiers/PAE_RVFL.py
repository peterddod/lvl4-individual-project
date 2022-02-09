import torch
from torch import nn
import numpy as np
from .AE_RVFL import AE_RVFL


class PAE_RVFL():  
    
    def __init__(self, input_size, h_size, out_size=None, subnets=1, device=None, r=(1,1), sc=0.5, sb=0.5):
        self._input_size = input_size
        self._h_size = h_size
        self._out_size = out_size
        self._subnets = subnets
        self._device = device
        self._rvfls = []
        self._r = r

        self._decision_weights = None

        if out_size != None:
            self._decision_weights = nn.init.uniform_(torch.empty(h_size, out_size, device=self._device), a=-r[0], b=r[0])

        # Initialise RVFL sub-networks
        for i in range(subnets):
            self._rvfls.append(AE_RVFL(
                self._input_size,
                self._h_size,
                self._device,
                r = self._r,
                sb=sb,
                sc=sc,
            ))

        self._activation = torch.relu
    
    def predict(self, X):
        rvfl_out = self._rvfls[0].predict(X)

        for i in range(1, self._subnets):
            rvfl_out = np.hstack([rvfl_out, self._rvfls[i].predict(X)])

            rvfl_out = torch.from_numpy(rvfl_out)

        if self._out_size != None:
            rvfl_out = rvfl_out.mm(self._decision_weights)

        return rvfl_out


    def train(self, X, y=None):
        rvfl_out = self._rvfls[0].train(X)

        for i in range(1, self._subnets):
            rvfl_out = np.hstack([rvfl_out, self._rvfls[i].train(X)])

            rvfl_out = torch.from_numpy(rvfl_out)

        if self._out_size != None:
            self._decision_weights = torch.pinverse(rvfl_out).mm(y)

        return rvfl_out               
