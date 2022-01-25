import torch
from torch import nn


class DOSELM():  
    
    def __init__(self, input_size, output_size, h_size, device=None):
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = output_size
        self._device = device

        self._weights = []
        self._biases = []

        self._datapoints = 0

        prev_layer_size = self._input_size

        for layer_size in self._h_size:
            self._weights.append(nn.init.uniform_(torch.empty(prev_layer_size, layer_size, device=self._device), a=-1., b=1.))
            self._biases.append(nn.init.uniform_(torch.empty(layer_size, device=self._device), a=-1., b=1.))
            prev_layer_size = layer_size
 
        self._beta = nn.init.uniform_(torch.empty(prev_layer_size, self._output_size, device=self._device), a=-1., b=1.)

        self._activation = torch.tanh
        self._activation_inverse = torch.atanh
    
    def predict(self, X):
        prev_out = X

        for i in range(len(self._h_size)):
            temp = prev_out.mm(self._weights[i])
            prev_out = self._activation(torch.add(temp, self._biases[i]))

        out = prev_out.mm(self._beta)

        return out

    def train(self, X, y):
        X.to(self._device)
        y.to(self._device)

        temp = X.mm(self._weights[0]).to(self._device)
        H = self._activation(torch.add(temp, self._biases[0])).to(self._device)

        for i in range(1, len(self._h_size)):
            temp = H.mm(self._weights[i]).to(self._device)
            H = self._activation(torch.add(temp, self._biases[i]))

        H_pinv = torch.pinverse(H)

        if self._datapoints == 0:
            self._datapoints += X.shape[0]
            self._beta = H_pinv.mm(y)

        else:
            self._datapoints += X.shape[0]
        
            beta_prime = H_pinv.mm(y)
            beta_diff = self._beta - beta_prime

            self._beta = self._beta - beta_diff*(X.shape[0]/self._datapoints)

            for i in range(1):
                beta_inv = torch.pinverse(self._beta)
                X_inv = torch.pinverse(X)

                # Calculate new biases for each layer
                a = X_inv.mm(self._activation_inverse(y.mm(beta_inv)) - self._biases[i])
                a_diff = self._weights[i] - a
                print(a_diff.size())
                self._weights[i] = self._weights[i] - a_diff*0.00001

                # Calculate new weights for each layer
                temp = X.mm(self._weights[i])
                temp2 = y.mm(beta_inv)
                b = self._activation_inverse(temp2) - temp
                b_diff = self._biases[i] - b
                self._biases[i] = self._biases[i] - torch.mean(b_diff, 0)*0.00001

            temp = X.mm(self._weights[0]).to(self._device)
            H = self._activation(torch.add(temp, self._biases[0])).to(self._device)

            for i in range(1, len(self._h_size)):
                temp = H.mm(self._weights[i]).to(self._device)
                H = self._activation(torch.add(temp, self._biases[i]))

            H_pinv = torch.pinverse(H)

            self._datapoints += X.shape[0]
            self._beta = H_pinv.mm(y)

            