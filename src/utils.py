import torch.nn as nn


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()
        # Layers


    def sweeping_build(self, input_size, output_size, hidden_size: list, n_layers, dropout): # TODO: add dropout
        layers = []
        in_size = input_size
        for i in range(n_layers): # adjusted for input layer
            out_size = hidden_size[i]
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU()) # TODO: we could use different activation functions
            in_size = out_size

        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)


    def add_configuration(self, network):
        self.net = network

    def forward(self, x):
        return self.net(x)