import numpy as np
import torch
from torch import nn, sqrt
from nitools.convolution import ResBlock, OrthoConv2D, SqrtPool2D
from nitools.architecture import Pipeline
from nitools.classifiers import PAE_ELM


class MiniRes():

    def __init__(self, in_channels=1, _lambda=0.1, a=0.1):
        self._classifier = None

        self._model = Pipeline(
            OrthoConv2D(in_channels=in_channels, out_channels=64, kernel_size=7, stride=1),
            nn.BatchNorm2d(64),
            SqrtPool2D(kernel_size=2, same=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(channels=64, rf=3, _lambda=_lambda, a=a),

            OrthoConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            SqrtPool2D(kernel_size=2, same=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(channels=128, rf=3, _lambda=_lambda, a=a),

            OrthoConv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            SqrtPool2D(kernel_size=2, same=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(channels=256, rf=3, _lambda=_lambda, a=a),

            OrthoConv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            SqrtPool2D(kernel_size=2, same=True),
            ResBlock(channels=512, rf=3, _lambda=_lambda, a=a),

            nn.AvgPool2d(kernel_size=4, stride=1),
            nn.Flatten(),
        )
    
    def predict(self, X):
        n = int(X.size()[0])
        
        X = self._model(X)
        f = int(np.prod(X.size())/n)
        C = torch.reshape(X, (n, f)).detach()

        y = self._classifier.predict(C)

        return y


    def train(self, X, y):
        n = int(X.size()[0])
        
        X = self._model.train(X, y)
        f = int(np.prod(X.size())/n)
        C = torch.reshape(X, (n, f)).detach()

        print(C.size())

        self._classifier = PAE_ELM(
            in_size=C.size()[1],
            h_size=600,
            out_size=10,
            subnets=3,
            )

        H = self._classifier.train(C, y) 