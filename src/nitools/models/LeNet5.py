import numpy as np
import torch
from torch import nn, sqrt
from torch.nn.utils.parametrizations import orthogonal
from ..architecture.Pipeline import Pipeline
from ..convolution.Conv2D import Conv2D
from torch.nn import Conv2d


class LeNet5():

    def __init__(self, classifier, in_channels=1, weight_train=True):
        self._classifier = classifier
        self._model = []

        if weight_train:
            self._model = Pipeline(
                Conv2D(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2, stride=1),   
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1), 
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
            )
        else:
            self._model = Pipeline(
                Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2, stride=1),   
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), 
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
            )
    
    def __getitem__(self, index):
        return self._model[index]

    def predict(self, X):
        n = int(X.size()[0])
        
        X = self._model(X)
        f = int(np.prod(X.size())/n)
        C = torch.reshape(X, (n, f)).detach()

        y = self._classifier.predict(C)

        return y


    def train(self, X, y):
        n = int(X.size()[0])
        
        X = self._model.train(X)
        f = int(np.prod(X.size())/n)
        C = torch.reshape(X, (n, f)).detach()

        H = self._classifier.train(C, y) 