# encoding: utf-8

import time
import torch
import itertools
import numpy as np
from PIL import Image
from grid_sample import GridSample
from torch.autograd import Variable
from thin_plate_spline import ThinPlateSpline

source_image = Image.open('source.jpg').convert(mode = 'RGB')
source_image = np.array(source_image).astype('float32')
source_image = np.expand_dims(source_image.swapaxes(2, 1).swapaxes(1, 0), 0)
source_image = Variable(torch.from_numpy(source_image))
_, _, H, W = source_image.size()

# creat control points
target_control_points = torch.Tensor(list(itertools.product(range(1, W + 1, W // 4), range(1, H + 1, H // 4))))
source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-20, 20)

print('initialize module')
beg_time = time.time()
tps = ThinPlateSpline(H, W, target_control_points)
past_time = time.time() - beg_time
print('initialization takes %.02fs' % past_time)

source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
X, Y = source_coordinate.view(1, H, W, 2).split(1, dim = 3)
X = X * 2 / W - 1
Y = Y * 2 / H - 1
grid = torch.cat([X, Y], dim = 3)
grid_sample = GridSample(H, W, 255)
target_image = grid_sample(source_image, grid)
target_image = target_image.data.numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2)
target_image = Image.fromarray(target_image.astype('uint8'))
target_image.save('target.jpg')
