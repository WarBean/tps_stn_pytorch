# encoding: utf-8

import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 10))
for path in sorted(glob.glob('accuracy_log/*.txt')):
    accu_list = [float(line.strip()) for line in open(path)]
    label = os.path.basename(path)[:-4]
    model1, model2, angle, grid = label.split('_')
    linestyle = {
        'angle60': '-',
        'angle90': ':',
    }[angle]
    color = {
        'no_stn_grid2': '#F08080',
        'no_stn_grid3': '#FF6347',
        'no_stn_grid4': '#FF4500',
        'no_stn_grid5': '#FF0000',
        'bounded_stn_grid2': '#98FB98',
        'bounded_stn_grid3': '#00FF7F',
        'bounded_stn_grid4': '#3CB371',
        'bounded_stn_grid5': '#2E8B57',
        'unbounded_stn_grid2': '#00BFFF',
        'unbounded_stn_grid3': '#0000FF',
        'unbounded_stn_grid4': '#0000CD',
        'unbounded_stn_grid5': '#000080',
    }[model1 + '_' + model2 + '_' + grid]
    plt.plot(
        list(range(len(accu_list))), accu_list,
        color = color, linestyle = linestyle, linewidth = 0.5,
        label = label,
    )
plt.legend(loc = 'lower right')
if not os.path.isdir('demo'):
    os.makedirs('demo')
plt.savefig('demo/curve.png')
