# encoding: utf-8

import os
import torch
import random
import argparse
import mnist_model
import data_loader
from PIL import Image, ImageDraw
from mnist_model import STNClsNet
from grid_sample import grid_sample
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 64)
parser.add_argument('--angle', type = int, default = 60)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--checkpoint_id', type = int, required = True)
parser.add_argument('--model', required = True)
parser.add_argument('--grid_size', type = int, default = 3)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

assert args.model in ['bounded_stn', 'unbounded_stn']
model = mnist_model.get_model(args)
model.load_state_dict(torch.load('checkpoint/%s_angle%d_%03d.pth' % (args.model, args.angle, args.checkpoint_id)))
test_loader = data_loader.get_test_loader(args)

if not os.path.isdir('image'):
    os.makedirs('image')

for batch_idx, (source_data, target) in enumerate(test_loader):
    print(batch_idx)
    if args.cuda:
        source_data = source_data.cuda()
    batch_size = source_data.size(0)
    source_data = Variable(source_data, volatile = True)
    source_control_points = model.loc_net(source_data)
    source_coordinate = model.tps(source_control_points)
    grid = source_coordinate.view(batch_size, 28, 28, 2)
    target_data = grid_sample(source_data, grid)

    source_data = (source_data.data[:, 0] * 255).cpu().numpy().astype('uint8')
    target_data = (target_data.data[:, 0] * 255).cpu().numpy().astype('uint8')
    for i in range(batch_size):
        # resize for better visualization
        source_image = Image.fromarray(source_data[i]).convert('RGB').resize((128, 128))
        target_image = Image.fromarray(target_data[i]).convert('RGB').resize((128, 128))
        # create grey canvas for external control points
        canvas = Image.new(mode = 'RGB', size = (64 * 7, 64 * 4), color = (128, 128, 128))
        canvas.paste(source_image, (64, 64))
        canvas.paste(target_image, (64 * 4, 64))
        source_points = source_control_points.data[i]
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
        canvas.save('image/%s_%04d.jpg' % (args.model, batch_idx * args.batch_size + i))
