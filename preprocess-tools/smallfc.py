# Max Van Gelder
# 7/28/20
# Small fully connected neural network to test feasibility of classifying with a CPU

import math
import torch.nn.functional as F
from torch import nn


class SmallFC(nn.Module):
    def __init__(self, input_dim, num_output_classes):
        super().__init__()

        # Pretty generic fully connected neural network of the form
        # input_dim X input_dim * .2 X input_dim * .15 X num_output_classes * 8 X num_output_classes
        self.fc1 = nn.Linear(input_dim, math.floor(input_dim * .1))
        self.fc2 = nn.Linear(math.floor(input_dim * .1), math.floor(input_dim * .05))
        self.fc3 = nn.Linear(math.floor((input_dim * .05)), math.ceil(num_output_classes * 8))
        self.fc4 = nn.Linear(math.ceil(num_output_classes * 8), num_output_classes)

    def forward(self, x):
        # TODO: This is pretty ugly. Clean it up.
        # Normal feedforward with ReLU nonlinearity
        return F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))))
