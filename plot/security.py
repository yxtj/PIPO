# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from .base import *

# %% data functions

def pick_data(fn):
    record = np.load(fn, allow_pickle=True)
    losses = record['loss_record']
    times = record['time_record']
    diffs = record['diff_record']
    mdiffs = diffs.mean(1)
    return losses, times, diffs, mdiffs


# %% draw functions

def draw_losses(losses, legend):
    assert len(losses) == len(legend)
    plt.figure()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.plot(losses.T / legend)
    plt.legend(legend)
    plt.tight_layout()


def draw_diffs(diffs, legend):
    assert len(diffs) == len(legend)
    plt.figure()
    plt.ylabel('absolute diff ratio')
    plt.xlabel('epoch')
    plt.plot(diffs.T)
    plt.legend(legend)
    plt.tight_layout()

def draw_diff_dis(diff, nbin=100, rng_r=(0.025,  0.975), rng=None, cdf=True, newfig=True):
    if newfig:
        plt.figure()
    # plt.hist(diff, nbin, density=True, cumulative=True, )
    if rng is None:
        rng = np.quantile(diff, rng_r)
        rng_r = (sum(diff<rng[0])/len(diff), sum(diff<rng[1])/len(diff))
    h, x = np.histogram(diff, nbin, rng)
    x = (x[1:] + x[:-1])/2
    w = x[1]-x[0]
    if cdf:
        # h = np.cumsum(h) / h.sum()
        h = np.cumsum(h) / h.sum() * (rng_r[1] - rng_r[0]) + rng_r[0]
        plt.plot(x, h)
    else:
        h = h/h.sum()
        plt.bar(x, h, w)
    plt.xlabel('diff ratio')
    if cdf:
        plt.ylabel('CDF')
    else:
        plt.ylabel('density')
    plt.tight_layout()

# %% draw

# losses, times, diffs, mdiffs = pick_data('minionn_1_10000_record.npz')
# plt.plot(losses)
# plt.plot(mdiffs)

legend = [1, 10, 100, 1000, 10000]

fnlist = [f'minionn_{i}_10000_record.npz' for i in legend]


loss_data = []
time_data = []
diff_data = []
for i, fn in enumerate(fnlist):
    l, t, d, _ = pick_data(fn)
    loss_data.append(l)
    time_data.append(t)
    diff_data.append(d)

loss_data = np.array(loss_data)
time_data = np.array(time_data)
diff_data = np.array(diff_data)
madiff_data = np.abs(diff_data).mean(2)
mdiff_data = diff_data.mean(2)


best_idx = np.argmin(madiff_data, 1)
best_diffs = np.array([ diff_data[i, best_idx[i]] for i in range(len(legend)) ])


draw_losses(loss_data, legend)
plt.legend([f'sampes: {i:.0e}' for i in legend])

draw_diffs(madiff_data, legend)
plt.legend([f'sampes: {i:.0e}' for i in legend], loc='upper right')
plt.yscale('log')


draw_diff_dis(best_diffs[2], cdf=True)

draw_diff_dis(best_diffs[2], cdf=False)

