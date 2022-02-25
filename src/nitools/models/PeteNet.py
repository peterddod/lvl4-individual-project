import numpy as np
import torch
from torch import nn, sqrt
from ..architecture.Pipeline import Pipeline
from ..convolution import ResConv2D
from torch.nn import Conv2d


class PeteNet():

    def __init__(self, classifier):
        self._classifier = classifier

        self._model = Pipeline(
            Conv2d(in_channels=1, out_channels=10, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   
            ResConv2D(channels=10, kernel_size=3, padding=1, stride=1), 
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
        )

    
    def predict(self, X):
        w_h = int(np.sqrt(X.size()[1]))
        n = int(X.size()[0])
        
        X = self._model(torch.reshape(X, (n, 1, w_h, w_h)))
        C = X.detach()

        y = self._classifier.predict(C)

        return y


    def train(self, X, y):
        w_h = int(np.sqrt(X.size()[1]))
        n = int(X.size()[0])
        
        X = self._model(torch.reshape(X, (n, 1, w_h, w_h)))
        C = X.detach()

        H = self._classifier.train(C, y) 