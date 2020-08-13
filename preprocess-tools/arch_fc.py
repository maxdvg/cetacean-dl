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
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
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


    # TODO: Don't hardcode cutoffs
    dataset = WhaleSongDataset(bh_cur, hb_cur, .65, .80)

    # Divide the data into mutually exclusive training and validation sets
    validation_size = 0.2
    num_data = len(dataset)
    idxs = list(range(num_data))
    np.random.shuffle(idxs)
    split = int(math.floor(validation_size * num_data))
    train_indices, val_indices = idxs[split:], idxs[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    batch_size = 4

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=0)

    classes = ('humpback', 'bowhead')

    # Instantiate the model
    # TODO: Don't hardcode input/output dimensions
    model = SmallFC(18382, 2)

    # TODO: Verify weight tensor is doing what you want it to do (stop network from biasing towards most common class)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1/dataset.num_hb, 1/dataset.num_bh]))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
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
            if i % 25 == 24:  # print every 25 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 25))
                running_loss = 0.0

            torch.save(model.state_dict(), "model.pt")

    print('Finished Training')

