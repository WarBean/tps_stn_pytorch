# encoding: utf-8

import os
import glob
import imageio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required = True)
parser.add_argument('--angle', type = int, default = 60)
parser.add_argument('--grid_size', type = int, default = 4)
args = parser.parse_args()

gif_dir = 'gif/%s_angle%d_grid%d/' % (args.model, args.angle, args.grid_size)
if not os.path.isdir(gif_dir):
    os.makedirs(gif_dir)

max_iter = 100
for i in range(max_iter):
    print('sample %d' % i)
    paths = sorted(glob.glob('image/%s_angle%d_grid%d/sample%03d_*.png' % (
        args.model, args.angle, args.grid_size, i,
    )))
    images = [imageio.imread(path) for path in paths]
    for _ in range(20): images.append(images[-1]) # delay at the end
    imageio.mimsave(gif_dir + 'sample%03d.gif' % i, images)
