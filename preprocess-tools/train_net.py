# Max Van Gelder
# 7/27/20

# Train a small fully connected neural network to differentiate
# humpback and bowhead song based only on the archipelago reconstruction

# Intentionally very small for CPU training
# argv[1] is path to bowhead database
# argv[2] is path to humpback database

import math
import numpy as np
import sqlite3
import sys
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import torchvision
import torchvision.transforms as transforms

from networks import SmallFC, ConvNet
from dldata import WhaleSongDataset


def print_time(message):
    print("{} {}".format(message, time.time()))


if __name__ == "__main__":
    # bowhead database connection
    bh_conn = sqlite3.connect(sys.argv[1])
    bh_cur = bh_conn.cursor()
    # humpback database connection
    hb_conn = sqlite3.connect(sys.argv[2])
    hb_cur = hb_conn.cursor()


    # TODO: Don't hardcode cutoffs
    dataset = WhaleSongDataset(bh_cur, hb_cur, .65, .80)

    # Divide the data into mutually exclusive training and validation sets
    validation_size = 0.1
    test_size = 0.1
    num_data = len(dataset)
    idxs = list(range(num_data))
    np.random.shuffle(idxs)
    val_split = int(math.floor(validation_size * num_data))
    train_split = int(math.floor(test_size * num_data)) + val_split
    train_indices, val_indices, test_indices = idxs[val_split:], idxs[val_split:train_split], idxs[train_split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    batch_size = 8

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0)

    classes = ('humpback', 'bowhead')

    # Instantiate the model
    # TODO: Don't hardcode input/output dimensions
    model = ConvNet()
    # model = SmallFC(18382, 2)

    # TODO: Verify weight tensor is doing what you want it to do (stop network from biasing towards most common class)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1/dataset.num_hb, 1/dataset.num_bh]))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 3

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'].float(), data['classification']

            # Reshape the input tensor so that it works for the convolutional net
            # The images are 182x101 and the batch size is 8
            inputs = torch.reshape(inputs, (8, 1, 182, 101))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 25 == 24:  # print every 25 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 25))
                running_loss = 0.0

        # Evaluate model on validation set
        model.eval()

        val_loss = 0.0
        for j, validation_data in enumerate(validation_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = validation_data['image'].float(), validation_data['classification']
            inputs = torch.reshape(inputs, (8, 1, 182, 101))

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss

        print("Val loss @ EP{}: ".format(epoch, val_loss / len(validation_data)))

        model.train()

        torch.save(model.state_dict(), "model.pt")

    print('Finished Training')

