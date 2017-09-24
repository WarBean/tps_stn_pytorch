# encoding: utf-8

import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from grid_sample import grid_sample
from torch.autograd import Variable
from thin_plate_spline import ThinPlateSpline

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

class LocNet(nn.Module):

    def __init__(self, grid_size):
        super(LocNet, self).__init__()
        self.cnn = CNN(grid_size ** 2 * 2)
        self.cnn.fc2.weight.data.mul_(0.001)
        control_points = torch.Tensor(list(itertools.product(
            torch.arange(-1.0, 1.00001, 2.0 / (grid_size - 1)),
            torch.arange(-1.0, 1.00001, 2.0 / (grid_size - 1)),
        )))
        self.register_buffer('control_points', control_points)

    def forward(self, x):
        batch_size = x.size(0)
        offset = F.tanh(self.cnn(x))
        points = Variable(self.control_points) + offset.view(batch_size, -1, 2)
        return points

class STNClsNet(nn.Module):

    def __init__(self):
        super(STNClsNet, self).__init__()
        self.loc_net = LocNet(4)
        self.cls_net = ClsNet()
        self.tps = ThinPlateSpline(28, 28, self.loc_net.control_points)

    def forward(self, x):
        batch_size = x.size(0)
        source_control_points = self.loc_net(x)
        source_coordinate = self.tps(source_control_points)
        grid = source_coordinate.view(batch_size, 28, 28, 2)
        transformed_x = grid_sample(x, grid)
        logit = self.cls_net(transformed_x)
        return logit
