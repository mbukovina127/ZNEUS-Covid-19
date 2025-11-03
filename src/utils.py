import numpy as np
import torch.nn as nn
import torch.optim as optim

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential()

    def forward(self, x):
        return self.net(x)
