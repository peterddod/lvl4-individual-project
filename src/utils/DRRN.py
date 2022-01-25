from sympy import rf
import torch
from torch import nn


class DRRN():  
    
    def __init__(self, input_size, output_size, h_size, subnets=1, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = output_size
        self._subnets = subnets
        self._device = device

        self._ae_weights = []
        self._ae_biases = []
        self._rvfl_weights = []
        self._rvfl_biases = []
        self._rvfl_betas = []
        self._rvfl_links = []

        prev_layer_size = self._input_size

        # 1. Randomly initialise RVFL sub-networks parameters
        for i in range(subnets):
            self._ae_weights.append(nn.init.uniform_(torch.empty(prev_layer_size ,h_size, device=self._device), a=-1., b=1.))
            self._ae_biases.append(nn.init.uniform_(torch.empty(h_size, device=self._device), a=-1., b=1.))
            self._rvfl_weights.append(nn.init.uniform_(torch.empty(prev_layer_size, h_size, device=self._device), a=-1., b=1.))
            self._rvfl_biases.append(nn.init.uniform_(torch.empty(h_size, device=self._device), a=-1., b=1.))
            self._rvfl_betas.append(nn.init.uniform_(torch.empty(h_size, device=self._device), a=-1., b=1.))
            self._rvfl_links.append(nn.init.uniform_(torch.empty(h_size, device=self._device), a=-1., b=1.))

            prev_layer_size = h_size

        # 2. Randomly initialise ELM decision sub-network parameters
        self._elm_weights = nn.init.uniform_(torch.empty(h_size, h_size, device=self._device), a=-1., b=1.)
        self._elm_biases = nn.init.uniform_(torch.empty(h_size, device=self._device), a=-1., b=1.)
        self._elm_beta = nn.init.uniform_(torch.empty(prev_layer_size, self._output_size, device=self._device), a=-1., b=1.)

        self._activation = torch.relu
    
    def predict(self, X):
        rvfl_out = X

        for i in range(self._subnets):
            # Pass data through RVFLs
            temp = rvfl_out.mm(self._rvfl_weights[i])
            H = self._activation(torch.add(temp, self._rvfl_biases[i]))
            rvfl_out = torch.add(rvfl_out.mm(self._rvfl_links[i]), H.mm(self._rvfl_betas[i]))

        # Make prediction with ELM
        temp = rvfl_out.mm(self._elm_weights)                        # input * weights
        H = self._activation(torch.add(temp, self._elm_biases))      # g(aX + b)                            
        out = H.mm(self._elm_beta)                                   # output weights

        return out

    def train(self, X, y):
        X.to(self._device)
        y.to(self._device)

        rvfl_out = X

        # 1. Create representation RVFL Networks
        for i in range(self._subnets):
            self._rvfl_links[i], ae_H = self._autoencode(rvfl_out, i)

            # 1b. Train RVFL  
            temp = rvfl_out.mm(self._rvfl_weights[i])                       # input * weights
            H = self._activation(torch.add(temp, self._ae_biases[i]))       # g(aX + b)
            H_pinv = torch.pinverse(H)                                      # H^-1
            self._rvfl_betas[i] = H_pinv.mm(ae_H)                           # output weights

            rvfl_out = torch.add(rvfl_out.mm(self._rvfl_links[i]), H.mm(self._rvfl_betas[i]))  # output from rvfl network

        # 2. Train ELM decision network
        temp = rvfl_out.mm(self._elm_weights)                           # input * weights
        H = self._activation(torch.add(temp, self._elm_biases))         # g(aX + b)
        H_pinv = torch.pinverse(H)                                      # H^-1
        self._elm_beta = H_pinv.mm(y)                                   # output weights
    
    def _autoencode(self, X, i):
        temp = X.mm(self._ae_weights[i])                                # input * weights
        ae_H = torch.tanh(torch.add(temp, self._ae_biases[i]))    # g(aX + b)
        ae_H_pinv = torch.pinverse(ae_H)                                # H^-1
        beta = ae_H_pinv.mm(X)                                          # output weights
        return torch.pinverse(beta), ae_H                               # beta^-1 = direct links