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
        # self.fc1 = nn.Linear(input_dim, math.floor(input_dim * .25))
        # self.fc2 = nn.Linear(math.floor(input_dim * .25), math.floor(input_dim * .1))
        # self.fc3 = nn.Linear(math.floor((input_dim * .1)), math.ceil(num_output_classes * .05))
        # self.fc4 = nn.Linear(math.ceil(num_output_classes * .05), num_output_classes)

        self.fc1 = nn.Linear(input_dim, math.floor(1000))
        self.fc2 = nn.Linear(math.floor(1000), math.floor(70))
        self.fc3 = nn.Linear(math.floor(70), math.ceil(num_output_classes * 8))
        self.fc4 = nn.Linear(math.ceil(num_output_classes * 8), num_output_classes)

    def forward(self, x):
        # Normal feedforward with ReLU nonlinearity
        return F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))))


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 input image channel, 15 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 15, 5)
        self.conv2 = nn.Conv2d(15, 8, 4)

        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        # Big max-pooling window, we really need to reduce dimensionality for the fully-connected layers!
        x = F.max_pool2d(x, 7)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @classmethod
    def num_flat_features(cls, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    print("You shouldn't be running this program!")