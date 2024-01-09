# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def gen_preimage_model(conv: nn.Conv2d):
    r = nn.ConvTranspose2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                           padding=conv.padding, bias=conv.bias is not None)
    w = torch.flip(conv.weight.data, [2,3])/conv.weight.data.sum()
    r.weight.data = w
    return r


@torch.no_grad()
def gradient(model, x, step):
    # assert x.shape == step.shape
    c, h, w = x.shape[-3:]
    res = torch.zeros_like(x)
    r = model(x)
    for i in range(c):
        for j in range(h):
            for k in range(w):
                xp = x.clone()
                xp[...,i,j,k] += step[i,j,k]
                rp = model(xp)
                res[i,j,k] = (rp-r)/step[i,j,k]
    return res


def check_layer(model, x, eps):
    pass


def binary_probe_cp_idx(model, start, step, zero_thr, f, l, df, dl):
    if torch.abs(df - dl) <= zero_thr:
        return []
    if f >= l:
        return [f] if torch.abs(df - dl) > zero_thr else []
    m = (f + l)//2
    mid = start + step*m
    dm = model(mid+step)-model(mid)
    # left
    if torch.abs(df - dm) > zero_thr: # df != dm:
        res1 = binary_probe_cp_idx(model, start, step, zero_thr, f, m, df, dm)
    else:
        res1 = []
    # right
    if torch.abs(dm - dl) > zero_thr: # dm != dl:
        res2 = binary_probe_cp_idx(model, start, step, zero_thr, m, l, dm, dl)
    else:
        res2 = []
    return res1 + res2
    

@torch.no_grad()
def probe_cp(model, start, end, eps, zero_thr=1e-8):
    dist = torch.dist(start, end)
    n = int(dist//eps)
    step = (end - start) / n
    
    df = model(start+step)-model(start)
    dl = model(end+step)-model(end)
    idx = binary_probe_cp_idx(model, start, step, zero_thr*eps, 0, n, df, dl)
    cps = [start+i*step for i in idx]
    return cps


def extract(model, start, end, eps=1e-6):
    # find critical points
    cps = probe_cp(model, start, end, eps)
    
    # identify layers
    
    # extract weights


# %% test

def __test_binary_probe__():
    m1 = nn.Sequential(nn.Conv(1,1,3), nn.ReLU())
    eps = 1e-4
    
    start = torch.rand(1,3,3)
    end = -start
    rs, re = m1(start), m1(end)
    print(rs, re)
    assert rs * re == 0.0 and rs + re > 0.0
    
    n = int(torch.dist(start, end)/eps)
    step = (end-start)/n
    
    a,b = start,end
    while torch.dist(a,b)>eps:
        ga=m1(a+step)-m1(a)
        gb=m1(b+step)-m1(b)
        if torch.abs(ga-gb)/eps<1e-6:
            print('found', ga/eps, gb/eps)
            break
        mid = (a+b)/2
        gm=m1(mid+step)-m1(mid)
        t = torch.tensor([torch.dist(start, a)//eps, torch.dist(start, mid)//eps, torch.dist(start, b)//eps, ga, gm, gb])
        print(t.tolist())
        if torch.abs(ga-gm)/eps<1e-6:
            a = mid+step
            print('right')
        else:
            b = mid-step
            print('left')
    
    print(gradient(m1, a, step))
    print(gradient(m1, b, step))


def __test__():
    m0 = nn.Conv2d(1,1,3)
    m1 = nn.Sequential(nn.Conv2d(1,1,3), nn.ReLU())
    m2 = nn.Sequential(nn.Conv2d(1,1,3), nn.ReLU(), nn.Conv2d(1, 1, 3))
    m3 = nn.Sequential(nn.Conv2d(1,4,3), nn.ReLU(), nn.Conv2d(4, 1, 3))

    eps = 1e-6
    r1 = extract(m1, eps)



