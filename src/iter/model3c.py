from torch.nn import Module
from torch import nn, reshape
from numpy import sqrt


class Model3c(Module):
    def __init__(self):
        super(Model3c, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, stride=1),   
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        w_h = int(sqrt(x.size()[1]))
        n = int(x.size()[0])
        return self.model(reshape(x, (n, 1, w_h, w_h)))