# encoding: utf-8

import os
import glob
import torch
import random
import argparse
import mnist_model
import data_loader
import numpy as np
from mnist_model import STNClsNet
from grid_sample import grid_sample
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 64)
parser.add_argument('--angle', type = int, default = 60)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--model', required = True)
parser.add_argument('--span_range', type = int, default = 0.9)
parser.add_argument('--grid_size', type = int, default = 4)
args = parser.parse_args()

args.span_range_height = args.span_range_width = args.span_range
args.grid_height = args.grid_width = args.grid_size
args.image_height = args.image_width = 28

args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(1024)

assert args.model in ['bounded_stn', 'unbounded_stn']
model = mnist_model.get_model(args)
if args.cuda:
    model.cuda()
image_dir = 'image/%s_angle%d_grid%d/' % (args.model, args.angle, args.grid_size)
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

test_loader = data_loader.get_test_loader(args)
target2data_list = { i: [] for i in range(10) }
total = 0
N = 10
for data_batch, target_batch in test_loader:
    for data, target in zip(data_batch, target_batch):
        data_list = target2data_list[target]
        if len(data_list) < N:
            data_list.append(data)
            total += 1
    if total == N * 10:
        break
data_list = [target2data_list[i][j] for i in range(10) for j in range(N)]
source_data = torch.stack(data_list)
if args.cuda:
    source_data = source_data.cuda()
batch_size = N * 10
frames_list = [[] for _ in range(batch_size)]

paths = sorted(glob.glob('checkpoint/%s_angle%d_grid%d/*.pth' % (
    args.model, args.angle, args.grid_size,
)))[::-1]
font = ImageFont.truetype('Comic Sans MS.ttf', 20)
for pi, path in enumerate(paths): # path index
    print('path %d/%d: %s' % (pi, len(paths), path))
    model.load_state_dict(torch.load(path))
    source_control_points = model.loc_net(Variable(source_data, volatile = True))
    source_coordinate = model.tps(source_control_points)
    grid = source_coordinate.view(batch_size, 28, 28, 2)
    target_data = grid_sample(source_data, grid).data

    source_array = (source_data[:, 0] * 255).cpu().numpy().astype('uint8')
    target_array = (target_data[:, 0] * 255).cpu().numpy().astype('uint8')
    for si in range(batch_size): # sample index
        # resize for better visualization
        source_image = Image.fromarray(source_array[si]).convert('RGB').resize((128, 128))
        target_image = Image.fromarray(target_array[si]).convert('RGB').resize((128, 128))
        # create grey canvas for external control points
        canvas = Image.new(mode = 'RGB', size = (64 * 7, 64 * 4), color = (128, 128, 128))
        canvas.paste(source_image, (64, 64))
        canvas.paste(target_image, (64 * 4, 64))
        source_points = source_control_points.data[si]
        source_points = (source_points + 1) / 2 * 128 + 64
        draw = ImageDraw.Draw(canvas)
        for x, y in source_points:
            draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill = (255, 0, 0))
        source_points = source_points.view(args.grid_size, args.grid_size, 2)
        for j in range(args.grid_size):
            for k in range(args.grid_size):
                x1, y1 = source_points[j, k]
                if j > 0: # connect to left
                    x2, y2 = source_points[j - 1, k]
                    draw.line((x1, y1, x2, y2), fill = (255, 0, 0))
                if k > 0: # connect to up
                    x2, y2 = source_points[j, k - 1]
                    draw.line((x1, y1, x2, y2), fill = (255, 0, 0))
        draw.text((10, 0), 'sample %03d, iter %03d' % (si, len(paths) - 1 - pi), fill = (255, 0, 0), font = font)
        canvas.save(image_dir + 'sample%03d_iter%03d.png' % (si, len(paths) - 1 - pi))
