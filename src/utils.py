import numpy as np
import torch.nn as nn
import torch.optim as optim


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        # Layers
        self.net = nn.Sequential(
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 21),
            nn.Tanh(),
            nn.Linear(21, 21),
            nn.ReLU(),
            nn.Linear(21, 2),
        )

    def add_net(self, network):
        self.net = network

    def add_module(self, name, nnModule):
        self.net.add_module(name, nnModule)

    def forward(self, x):
        return self.net(x)
