import numpy as np
import torch
from torch import nn, sqrt
from ..architecture.Pipeline import Pipeline
from ..convolution.Conv2D import Conv2D
from torch.nn import Conv2d


class LeNet5():

    def __init__(self, classifier, weight_train=True):
        self._classifier = classifier
        self._model = []

        if weight_train:
            self._model = Pipeline(
                Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1),   
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
                Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1),   
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), 
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
            )
    
    def predict(self, X):
        w_h = int(np.sqrt(X.size()[1]))
        n = int(X.size()[0])
        
        X = self._model(torch.reshape(X, (n, 1, w_h, w_h)))
        C = torch.reshape(X, (n, 400)).detach()

        y = self._classifier.predict(C)

        return y


    def train(self, X, y):
        w_h = int(np.sqrt(X.size()[1]))
        n = int(X.size()[0])
        
        X = self._model.train(torch.reshape(X, (n, 1, w_h, w_h)))
        C = torch.reshape(X, (n, 400)).detach()

        H = self._classifier.train(C, y) 