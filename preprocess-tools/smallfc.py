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
        # TODO: This is pretty ugly. Clean it up.
        # Normal feedforward with ReLU nonlinearity
        return F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))))


class BasicBlock(nn.Module):
   expansion = 1


   def __init__(self, in_planes, planes, stride=1):
       super(BasicBlock, self).__init__()
       self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(planes)
       self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2 = nn.BatchNorm2d(planes)


       self.shortcut = nn.Sequential()
       if stride != 1 or in_planes != self.expansion*planes:
           self.shortcut = nn.Sequential(
               nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(self.expansion*planes))

class ResNet(nn.Module):
   def __init__(self, block, num_blocks, num_classes=2):
       super(ResNet, self).__init__()
       self.in_planes = 64


       self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(64)
       self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
       self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
       self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
       self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
       self.linear = nn.Linear(512*block.expansion, num_classes)


   def _make_layer(self, block, planes, num_blocks, stride):
       strides = [stride] + [1]*(num_blocks-1)
       layers = []
       for stride in strides:
           layers.append(block(self.in_planes, planes, stride))
           self.in_planes = planes * block.expansion
       return nn.Sequential(*layers)