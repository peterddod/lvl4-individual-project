import numpy as np
import torch
from torch import nn, sqrt
from torch.nn.utils.parametrizations import orthogonal
from nitools.architecture import Pipeline
from nitools.convolution import OrthoConv2D, SqrtPool2D, TargetCombNode
from nitools.classifiers import PAE_ELM


class R2_LRF_ELM():

    def __init__(self, in_channels=1, a=0.1):
        self._classifier = None
        self._model = []

        self._model = Pipeline(
            OrthoConv2D(in_channels=in_channels, out_channels=32, kernel_size=4, padding=0, stride=1),
            SqrtPool2D(kernel_size=2, same=True),
            TargetCombNode(a=a),
            nn.MaxPool2d(kernel_size=2,stride=2),
            OrthoConv2D(in_channels=1, out_channels=32, kernel_size=4, padding=0, stride=1),
            SqrtPool2D(kernel_size=2, same=True),
            TargetCombNode(a=a),
            nn.MaxPool2d(kernel_size=2,stride=2),
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