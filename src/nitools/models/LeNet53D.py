import numpy as np
import torch
from torch import nn, sqrt
from ..architecture.Pipeline import Pipeline
from torch.nn import Conv2d


class LeNet53D():

    def __init__(self, classifier):
        self._classifier = classifier
        self._model = []

        self._model = Pipeline(
            Conv2d(in_channels=3, out_channels=30, kernel_size=7, padding=3, stride=2),   
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=30, out_channels=48, kernel_size=3, stride=1), 
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
    
    def predict(self, X):
        w_h = int(np.sqrt(X.size()[1]))
        n = int(X.size()[0])
        
        X = self._model(torch.reshape(X, (n, 3, 32, 32)))
        C = torch.reshape(X, (n, 432)).detach()

        y = self._classifier.predict(C)

        return y


    def train(self, X, y):
        w_h = int(np.sqrt(X.size()[1]))
        n = int(X.size()[0])
        
        X = self._model.train(torch.reshape(X, (n, 3, 32, 32)))
        C = torch.reshape(X, (n, 432)).detach()

        H = self._classifier.train(C, y) 