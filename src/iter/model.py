from torch.nn import Module
from torch import nn, reshape
from numpy import sqrt


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=2, stride=1),   
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=48, kernel_size=3, stride=1, padding=2), 
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(240),
            nn.ReLU(),
            nn.Linear(240, 172),
            nn.ReLU(),
            nn.Linear(172, 10),
        )

    def __getitem__(self, index):
        return self.model[index]

    def forward(self, x):
        w_h = int(sqrt(x.size()[1]))
        n = int(x.size()[0])
        return self.model(reshape(x, (n, 1, w_h, w_h)))