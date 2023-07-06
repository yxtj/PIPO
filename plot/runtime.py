# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from plot.base import *

# %% plot function

def show_group_bar(data, names, ylabel, width=0.7,
                   logscale=False, showvalue=False):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    n, m = data.shape
    assert m == 2
    assert n == len(names)
    plt.figure()
    wb = width/m
    x = np.arange(n)
    bc1 = plt.bar(x - wb/2, data[:, 0], wb, label='offline', hatch='///')
    bc2 = plt.bar(x + wb/2, data[:, 1], wb, label='online', hatch='\\\\\\')
    plt.xticks(x, names)
    if showvalue:
        fs = plt.rcParams['font.size'] - 2
        plt.bar_label(bc1, data[:, 0], fontsize=fs)
        plt.bar_label(bc2, data[:, 1], fontsize=fs)
    if logscale:
        plt.yscale('log')
    # plt.legend(['offline', 'online'])
    # plt.legend()
    plt.legend(borderpad=0.3, handletextpad=0.2, columnspacing=1)
    plt.ylabel(ylabel)
    plt.tight_layout()


def show_stack_bar(data, names, ylabel, width=0.7,
                   logscale=False, showvalue=False):
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    # plt.bar(x, y, fill=False, hatch='///')
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    n, m = data.shape
    assert m == 2
    assert n == len(names)
    plt.figure()
    bc1 = plt.bar(range(n), data[:, 0], width, 0, label='offline', hatch='///')
    bc2 = plt.bar(range(n), data[:, 1], width, data[:, 0], label='online', hatch='\\\\\\')
    plt.xticks(range(n), names)
    if showvalue:
        fs = plt.rcParams['font.size'] - 2
        bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad=0.5)
        plt.bar_label(bc1, data[:, 0], fontsize=fs, padding=-fs-2, bbox=bbox)
        plt.bar_label(bc2, data[:, 1], fontsize=fs)
    if logscale:
        plt.yscale('log')
    # plt.legend(['offline', 'online'])
    plt.legend()
    plt.ylabel(ylabel)
    plt.tight_layout()


def show_stack_bar_group(data, names, groups, ylabel, width=0.8,
                         logscale=False, showvalue=False):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    g, n, m = data.shape
    assert g == len(groups)
    assert n == len(names)
    assert m == 2
    if showvalue:
        fs = plt.rcParams['font.size'] - 2
        bbox = dict(facecolor='white', alpha=0.8, edgecolor='white', pad=0.5)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure()
    x = np.arange(g)
    wb = width/n
    offset = -width/2 + wb/2
    for i in range(n):
        nm = names[i]
        bc1 = plt.bar(x + offset + wb*i, data[:, i, 0], wb, 0,
                      color=colors[i], edgecolor='black', label=nm+' off.', hatch='///')
        bc2 = plt.bar(x + offset + wb*i, data[:, i, 1], wb, data[:, i, 0],
                      color=colors[i], edgecolor='black', label=nm+' on.', hatch='\\\\\\')
        if showvalue:
            plt.bar_label(bc1, data[i, :, 0], fontsize=fs, padding=-fs-2, bbox=bbox)
            plt.bar_label(bc2, data[i, :, 1], fontsize=fs)
    if logscale:
        plt.yscale('log')
    plt.xticks(x, groups)
    plt.ylabel(ylabel)
    fs = plt.rcParams['font.size'] - 3
    plt.legend(ncol=2, fontsize=fs, borderpad=0.3, handletextpad=0.2, columnspacing=1)
    plt.tight_layout()


# %% draw for activation

# act_name = ['Local', 'Scaling\n(FaSeNet)', 'GC\n(Delphi)', 'Beaver\n(Delphi)']

act_name, act_time = load_time('activation.csv')
act_name, act_comm = load_comm('activation.csv')
act_name = break_names(act_name)

ylbl_time = 'Total execution time (s)'
ylbl_tomm = 'Total data transferred (MB)'

# show_stack_bar(act_time, act_name, ylbl_time, 0.7, False)
show_stack_bar(act_time, act_name, ylbl_time, 0.7, True, True)
plt.ylim(None, 200)

show_stack_bar(act_time[1:], act_name[1:], ylbl_time, 0.6, True, False)
show_stack_bar(act_time[1:], act_name[1:], ylbl_time, 0.6, True, True)
plt.ylim(None, 200)

show_group_bar(act_time[1:], act_name[1:], 'Execute time (s)', 0.8, True, True)
plt.ylim(None, 100)
plt.text(-0.2, 0.008, '0', fontsize=plt.rcParams['font.size']-2)

plt.ylim(None, 150)
plt.text(-0.2, 0.002, '0', fontsize=plt.rcParams['font.size']-2)

# communication

# show_stack_bar(act_comm, act_name, ylbl_tomm, 0.7, False)
show_stack_bar(act_comm, act_name, ylbl_tomm, 0.7, True, True)
plt.ylim(None, 20)

show_stack_bar(act_comm[1:], act_name[1:], ylbl_tomm, 0.6, True, False)
show_stack_bar(act_comm[1:], act_name[1:], ylbl_tomm, 0.6, True, True)
plt.ylim(None, 15)

show_group_bar(act_comm[1:], act_name[1:], 'Data transferred (MB)', 0.8, True, True)
plt.ylim(None, 15)
plt.text(-0.2, 0.0017, '0', fontsize=plt.rcParams['font.size']-2)
plt.text(0.2, 0.0017, '0', fontsize=plt.rcParams['font.size']-2)

# %% draw for pooling

# pool_name = ['max', 'avg']
# pool_name = ['max-pool', 'avg-pool']
# pool_name = ['max pool', 'avg pool']

pool_name, pool_time = load_time('pooling.csv')
pool_name, pool_comm = load_comm('pooling.csv')

ylbl_time = 'Total execution time (s)'
ylbl_tomm = 'Total data transferred (MB)'


show_stack_bar(pool_time, pool_name, ylbl_time, 0.6)
plt.legend(loc='lower right')
plt.xlim(-0.5, 1.5)

show_stack_bar(pool_time, pool_name, ylbl_time, 0.6, showvalue=True)
plt.legend(loc='lower right')
plt.ylim(0, 45)
plt.xlim(-0.5, 1.5)

show_group_bar(pool_time, pool_name, 'Execution time (s)', 0.7, showvalue=True)
plt.ylim(None, 32)
plt.xlim(-0.5, 1.5)

# communication cost

show_stack_bar(pool_comm, pool_name, ylbl_tomm, 0.6, True)

show_group_bar(pool_comm, pool_name, 'Data transferred (MB)', 0.7, True, showvalue=True)
plt.ylim(None, 5000)
plt.xlim(-0.5, 1.5)
plt.legend(loc='center right')


# %% compare model/system

# names = ['FaSeNet', 'Delphi', 'Delphi (ReLU)', 'Gazzele']
# groups = ['MiniONN', 'ResNet32']

names, time_minionn = load_time('minionn.csv')
names, time_resnet = load_time('resnet.csv')
names, comm_minionn = load_comm('minionn.csv')
names, comm_resnet = load_comm('resnet.csv')
groups = ['MiniONN', 'ResNet32']

show_stack_bar_group([time_minionn, time_resnet], names, groups, 'Total execution time (s)', 0.8)

show_stack_bar_group(np.array([comm_minionn, comm_resnet])/1000, names, groups, 'Total data transferred (GB)', 0.8)


show_stack_bar_group([time_minionn[:-1], time_resnet[:-1]], names[:-1], groups, 'Total execution time (s)', 0.8)

show_stack_bar_group(np.array([comm_minionn[:-1], comm_resnet[:-1]])/1000, names[:-1], groups, 'Total data transferred (GB)', 0.8)

# %% resnet family

names, time_rns = load_time('plot/resnets.csv')
names, comm_rns = load_comm('plot/resnets.csv')

show_group_bar(time_rns, names, 'Execution time (s)', 0.7, True)
plt.xticks(np.arange(6), names, rotation=-20)
plt.ylim(0.01, 8000)
plt.tight_layout()


show_group_bar(time_rns, names, 'Data transferred (MB)', 0.7, True)
plt.xticks(np.arange(6), names, rotation=-20)
plt.ylim(0.7, 50000)
plt.tight_layout()

