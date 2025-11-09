import torch.nn as nn

class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()

    def sweeping_build(self, input_size, output_size, hidden_size: list, activation_f, n_layers, dropout): # TODO: add dropout
        """
        Builds neural network from inputed parameters. Used for hyperparameter sweep
        :param input_size:
        :param output_size:
        :param hidden_size:
        :param activation_f:
        :param n_layers:
        :param dropout:
        """
        layers = []
        in_size = input_size
        for i in range(0, n_layers): # adjusted for input layer
            out_size = hidden_size[i]
            layers.append(nn.Linear(in_size, out_size))
            if activation_f == 'relu': layers.append(nn.ReLU())
            if activation_f == 'tanh': layers.append(nn.Tanh())
            # if i == n_layers - 1: layers.append(nn.Linear(in_size, output_size)) # adding dropout layer before last linear layer
            layers.append(nn.Dropout(dropout)) # adding dropout layer before last linear layer
            in_size = out_size

        layers.append(nn.Linear(in_size, output_size))
        self.net = nn.Sequential(*layers)


    def add_configuration(self, network):
        self.net = network

    def forward(self, x):
        return self.net(x)