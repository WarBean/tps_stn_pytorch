# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GridSample(nn.Module):

    def __init__(self, height, width, padding):
        super(GridSample, self).__init__()
        self.register_buffer('input_mask', torch.ones(1, 1, height, width))
        self.register_buffer('padding', torch.Tensor(1, 1, height, width).fill_(padding))

    def forward(self, input, grid):
        output = F.grid_sample(input, grid)
        output_mask = F.grid_sample(Variable(self.input_mask), grid)
        padded_output = output * output_mask + Variable(self.padding) * (1 - output_mask)
        return padded_output
