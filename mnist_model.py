# encoding: utf-8

import math
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from grid_sample import grid_sample
from torch.autograd import Variable
from tps_grid_gen import TPSGridGen

class CNN(nn.Module):
    def __init__(self, num_output):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_output)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class ClsNet(nn.Module):

    def __init__(self):
        super(ClsNet, self).__init__()
        self.cnn = CNN(10)

    def forward(self, x):
        return F.log_softmax(self.cnn(x))

def get_control_points(grid_size):
    return np.array(list(itertools.product(
        np.arange(-1.0, 1.00001, 2.0 / (grid_size - 1)),
        np.arange(-1.0, 1.00001, 2.0 / (grid_size - 1)),
    )))

class BoundedGridLocNet(nn.Module):

    def __init__(self, grid_size):
        super(BoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_size ** 2 * 2)

        control_points = get_control_points(grid_size).clip(-0.999, 0.999)
        bias = np.arctanh(control_points)
        bias = torch.from_numpy(bias).view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = F.tanh(self.cnn(x))
        return points.view(batch_size, -1, 2)

class UnBoundedGridLocNet(nn.Module):

    def __init__(self, grid_size):
        super(UnBoundedGridLocNet, self).__init__()
        self.cnn = CNN(grid_size ** 2 * 2)

        control_points = get_control_points(grid_size)
        bias = torch.from_numpy(control_points).view(-1)
        self.cnn.fc2.bias.data.copy_(bias)
        self.cnn.fc2.weight.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        points = self.cnn(x)
        return points.view(batch_size, -1, 2)

class STNClsNet(nn.Module):

    def __init__(self, args):
        super(STNClsNet, self).__init__()
        GridLocNet = {
            'unbounded_stn': UnBoundedGridLocNet,
            'bounded_stn': BoundedGridLocNet,
        }[args.model]
        self.loc_net = GridLocNet(args.grid_size)
        self.cls_net = ClsNet()
        self.tps = TPSGridGen(28, 28, torch.from_numpy(get_control_points(args.grid_size)))

    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, 28, 28, 2)
        transformed_x = grid_sample(x, grid)
        logit = self.cls_net(transformed_x)
        return logit

def get_model(args):
    if args.model == 'no_stn':
        print('create model without STN')
        model = ClsNet()
    else:
        print('create model with STN')
        model = STNClsNet(args)
    return model
