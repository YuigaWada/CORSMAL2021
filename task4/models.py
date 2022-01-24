from operator import xor
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)

        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 3),
                               padding=1,
                               stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(64)

        self.linear1 = nn.Linear(7 * 7 * 128, 64)
        self.linear2 = nn.Linear(64, 6)
        self.linear3 = nn.Linear(9, 1)

        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, img, dimension_vector):
        x = self.conv1(img)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = self.pool(x)

        x = x.view(-1, 7 * 7 * 128)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.bn5(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = torch.cat((x, dimension_vector), dim=1)
        x = self.linear3(x)

        return x
