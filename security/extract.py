# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


@torch.no_grad()
def gradient(model, x, step):
    # assert x.shape == step.shape
    c, h, w = x.shape[-3:]
    res = torch.zeros_like(x)
    r = model(x)
    for i in range(c):
        for j in range(h):
            for k in range(w):
                xp = x + step[i,j,k]
                rp = model(xp)
                res[i,j,k] = (rp-r)/step[i,j,k]
    return res


def check_layer(model, x, eps):
    pass


def binary_prob(model, f, l, step):
    pass

@torch.no_grad()
def probe(model, start, end, eps):
    dist = torch.dist(start, end)
    n = int(dist//eps)
    step = (end - start) / n
    res = []
    i = 0
    ds = model(start+eps)-model(start)
    de = model(end+eps)-model(end)
    while i<n:
        if torch.abs(ds - de)<1e-8:
            break
        mid = start + step*(n//2)



    return res


def extract(model, eps=1e-6):
    pass


# %% test

def __test__():
    m0 = nn.Conv2d(1,1,3)
    m1 = nn.Sequential(nn.Conv2d(1,1,3), nn.ReLU())
    m2 = nn.Sequential(nn.Conv2d(1,1,3), nn.ReLU(), nn.Conv2d(1, 1, 3))
    m3 = nn.Sequential(nn.Conv2d(1,4,3), nn.ReLU(), nn.Conv2d(4, 1, 3))

    eps = 1e-6
    r1 = extract(m1, eps)



