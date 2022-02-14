import numpy as np
import torch
from torch import nn, sqrt
from ..architecture.Pipeline import Pipeline
from ..convolution.Conv2D import Conv2D
from ..convolution.ResConv2D import ResConv2D


class ResNet18():

    def __init__(self, classifier, weight_train=True):
        self._classifier = classifier

        self._model = Pipeline(
            Conv2D(in_channels=1, out_channels=64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResConv2D(in_channels=1, out_channels=64, kernel_size=3),
            ResConv2D(in_channels=64, out_channels=128, kernel_size=3),
            ResConv2D(in_channels=128, out_channels=256, kernel_size=3),
            ResConv2D(in_channels=256, out_channels=512, kernel_size=3),
            nn.AvgPool2d(kernel_size=7),
            nn.Flatten(),
        )
    
    def predict(self, X):
        w_h = int(np.sqrt(X.size()[1]))
        n = int(X.size()[0])
        
        X = self._model(torch.reshape(X, (n, 1, w_h, w_h)))
        C = torch.reshape(X, (n, w_h**2)).detach()

        y = self._classifier.predict(C)

        return y


    def train(self, X, y):
        w_h = int(np.sqrt(X.size()[1]))
        n = int(X.size()[0])
        
        X = self._model.train(torch.reshape(X, (n, 1, w_h, w_h)))
        C = torch.reshape(X, (n, w_h**2)).detach()

        H = self._classifier.train(C, y) 