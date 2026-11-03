# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .base import *

# %% plot function

def show_acc_bar(data, names, width=0.7, rotation=0, showvalue=False):
    assert len(data) == len(names)
    plt.figure()
    #plt.bar(range(len(data)), data, tick_label=names, width=width)
    bct = plt.bar(range(len(data)), data, width=width)
    if showvalue:
        plt.bar_label(bct, padding=2)
    plt.xticks(range(len(data)), names, rotation=rotation)
    plt.ylabel("Accuracy (%)")
    plt.ylim((0, None))
    plt.tight_layout()


def show_acc_bar_group(data, names, legend, width=0.7, whiteedge=False, showvalue=False):
    if isinstance(data, list):
        data = np.array(data)
    assert data.ndim == 2
    n, m = data.shape
    assert n == len(names)
    assert m == len(legend)
    plt.figure()
    wb = width/m
    offset = -width/2 + wb/2
    bcts = []
    for i in range(m):
        b = plt.bar(np.arange(n)+offset+wb*i, data[:,i], width=wb, label=legend[i],
                    edgecolor='white', linewidth=0.5)
        bcts.append(b)
    if showvalue:
        for b in bcts:
            plt.bar_label(b, padding=3)
    plt.xticks(range(n), names)
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()


# %% draw accuracy for activation

# act_name = ['local', 'FaSeNet', 'Delphi-GC', 'Delphi-Beaver']
# act_name = ['Local', 'Scaling\nFaSeNet', 'GC\nDelphi', 'Beaver\nDelphi']
# act_name = ['Local', 'Scaling\n(FaSeNet)', 'GC\n(Delphi)', 'Beaver\n(Delphi)']

act_name, act_acc = load_acc('activation.csv')
act_name = break_names(act_name)

# show_acc_bar(act_acc, act_name, 0.7, 0, False)
# plt.ylim(0, 85)
show_acc_bar(act_acc, act_name, 0.7, 0, True)
plt.ylim(0, 90)

# show_acc_bar(act_acc[1:], act_name[1:], 0.7, 0, False)
# plt.ylim(0, 85)
show_acc_bar(act_acc[1:], act_name[1:], 0.6, 0, True)
plt.xlim(-0.5,2.5)
plt.ylim(0, 90)


# %% draw accuracy for pooling

# pool_name = ['max', 'avg']
# pool_name = ['max pool', 'avg pool']

pool_name, pool_acc = load_acc('pool.csv')

show_acc_bar(pool_acc, pool_name, 0.6, 0, True)
plt.ylim(0,100)
plt.xlim(-0.5, 1.5)
plt.tight_layout()

show_acc_bar(pool_acc, pool_name, 0.6, 0, True)
plt.ylim(50,100)
plt.xlim(-0.5, 1.5)
plt.tight_layout()

# %% compare model/system

# names = ['MiniONN', 'ResNet32']
# legend = ['FaSeNet', 'Delphi', 'Delphi (ReLU)', 'Gazelle']

legend, a1 = load_acc('minionn.csv')
legend, a2 = load_acc('resnet.csv')
data = [a1[1], a2[1]]
names = ['MiniONN', 'ResNet32']


show_acc_bar_group(data, names, legend, 0.7, True, False)
plt.legend(ncol=1, borderpad=0.3, handletextpad=0.2, columnspacing=1, loc='lower right')
plt.ylim(0, 100)
plt.tight_layout()


