# encoding: utf-8

import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize = (12, 12))
for path in sorted(glob.glob('accuracy_log/*.txt')):
    accu_list = [float(line.strip()) for line in open(path)]
    label = os.path.basename(path)[:-4]
    plt.plot(list(range(len(accu_list))), accu_list, label = label)
plt.legend(loc = 'lower right')
plt.savefig('accuracy_log/curve.png')
