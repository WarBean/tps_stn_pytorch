# encoding: utf-8

import time
import torch
import itertools
import numpy as np
from PIL import Image
from grid_sample import grid_sample
from torch.autograd import Variable
from tps_grid_gen import TPSGridGen

source_image = Image.open('demo/source_avatar.jpg').convert(mode = 'RGB')
source_image = np.array(source_image).astype('float32')
source_image = np.expand_dims(source_image.swapaxes(2, 1).swapaxes(1, 0), 0)
source_image = Variable(torch.from_numpy(source_image))
_, _, source_height, source_width = source_image.size()
target_height = 400
target_width = 400

# creat control points
target_control_points = torch.Tensor(list(itertools.product(
    torch.arange(-1.0, 1.00001, 2.0 / 4),
    torch.arange(-1.0, 1.00001, 2.0 / 4),
)))
source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1)

print('initialize module')
beg_time = time.time()
tps = TPSGridGen(target_height, target_width, target_control_points)
past_time = time.time() - beg_time
print('initialization takes %.02fs' % past_time)

source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
grid = source_coordinate.view(1, target_height, target_width, 2)
canvas = Variable(torch.Tensor(1, 3, target_height, target_width).fill_(255))
target_image = grid_sample(source_image, grid, canvas)
target_image = target_image.data.numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2)
target_image = Image.fromarray(target_image.astype('uint8'))
target_image.save('demo/target_avatar.jpg')
