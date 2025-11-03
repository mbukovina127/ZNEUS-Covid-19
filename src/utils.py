from collections import OrderedDict
import numpy as np
import torch.nn as nn
import torch.optim as optim

class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        # Layers
        self.net = nn.Sequential()


    def sweeping_build(self, input_size, output_size, hidden_size: list, n_layers, dropout): # TODO: add dropout
        layers = []
        in_size = input_size
        for i in range(n_layers+1): # adjusted for input layer
            out_size = hidden_size[i]
            layers.append((f"linear_{i}", nn.Linear(in_size, out_size)))
            layers.append((f"relu_{i}", nn.ReLU())) # TODO: we could use different activation functions
            in_size = out_size

        layers.append(("output", nn.Linear(in_size, output_size)))
        self.net = nn.Sequential(OrderedDict(layers))


    def add_configuration(self, network):
        self.net = network

    def add_module(self, name, nnModule):
        self.net.add_module(name, nnModule)

    def forward(self, x):
        return self.net(x)