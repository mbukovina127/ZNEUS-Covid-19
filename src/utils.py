import numpy as np
import torch.nn as nn
import torch.optim as optim


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        # Layers
        self.net = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def add_configuration(self, network):
        self.net = network

    def add_module(self, name, nnModule):
        self.net.add_module(name, nnModule)

    def forward(self, x):
        return self.net(x)