# Max Van Gelder
# 7/27/20

# Train a small fully connected neural network to differentiate
# humpback and bowhead song based only on the archipelago reconstruction

# Intentionally very small for CPU training

import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torchvision
import torchvision.transforms as transforms
from smallfc import SmallFC

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifar10data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Instantiate the model
    model = SmallFC(30, 3)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

