# Max Van Gelder
# 7/27/20

# Train a small fully connected neural network to differentiate
# humpback and bowhead song based only on the archipelago reconstruction

# Intentionally very small for CPU training
# argv[1] is path to bowhead database
# argv[2] is path to humpback database

import math
import sqlite3
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torchvision
import torchvision.transforms as transforms

from smallfc import SmallFC
from dldata import WhaleSongDataset

if __name__ == "__main__":
    # bowhead database connection
    bh_conn = sqlite3.connect(sys.argv[1])
    bh_cur = bh_conn.cursor()
    # humpback database connection
    hb_conn = sqlite3.connect(sys.argv[2])
    hb_cur = hb_conn.cursor()

    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # TODO: Don't hardcode cutoffs
    trainset = WhaleSongDataset(bh_cur, hb_cur, .405, .80)
    # TODO: Change batch size back to 4
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                              shuffle=True, num_workers=0)

    # testset = WhaleSongDataset(bh_cur, hb_cur, .405, .80)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                          shuffle=False, num_workers=2)

    classes = ('humpback', 'bowhead')


    # Instantiate the model
    # TODO: Don't hardcode input/output dimensions
    model = SmallFC(18382, 2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'].float(), data['classification']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 3 == 2:  # print every 3 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 3))
                running_loss = 0.0

    print('Finished Training')

