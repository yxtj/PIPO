# -*- coding: utf-8 -*-
import numpy as np
import math
import re

# %% models

model1=[(10,3), (1,3)]
model2=[(32,3), (16,3), (1,3)]
model3=[(16,5), (16,5), (1,5)]
model4=[(16,3), (32,3), (32,3), (1,3)]
model5=[(10,3), (10,3), (10,3), (10,3), (1,3)]
model6=[(6,5), (6,5), (16,5), (16,5), (1,12)] # mnist

# '3x32x32, 6c5-p2-16c5-p2-120c5-84c1-1c1'
model_lenet=[(6,5), 2, (16,5), 2, (1,5)]
# '3x227x227, 96c11-p4-p2-256c5-p2-384c3-384c3-384c3-p2-9216c1-4096c1-4096c1-1c1'
model_alex=[(96,11),4,2, (256,5),2, (384,3),(384,3),(384,3),2, (9216,1), (4096,1),(4096,1),(1,1)]

model_vgg11=[(64,3)] +[2]+ [(128,3)]  +[2]+ [(256,3)]*2 +[2]+ [(512,3)]*2 +[2]+ [(512,3)]*2 +[2,(4096,1),(1,1)]
model_vgg13=[(64,3)]*2 +[2]+ [(128,3)]*2  +[2]+ [(256,3)]*2 +[2]+ [(512,3)]*2 +[2]+ [(512,3)]*2 +[2,(4096,1),(1,1)]
model_vgg16=[(64,3)]*2 +[2]+ [(128,3)]*2  +[2]+ [(256,3)]*3 +[2]+ [(512,3)]*3 +[2]+ [(512,3)]*3 +[2,(4096,1),(1,1)]
model_vgg19=[(64,3)]*2 +[2]+ [(128,3)]*2  +[2]+ [(256,3)]*4 +[2]+ [(512,3)]*4 +[2]+ [(512,3)]*4 +[2,(4096,1),(1,1)]

model_res18=[(64,7), 2] + [(64,3)]*2 +[2]+ [(128,3)]*2 +[2]+ [(256,3)]*2 +[2]+ [(512,3)]*2 + [(1000,7), (1,1)]
model_res34=[(64,7), 2] + [(64,3)]*3 +[2]+ [(128,3)]*4 +[2]+ [(256,3)]*6 +[2]+ [(512,3)]*3 + [(1000,7), (1,1)]
model_res50=[(64,7), 2] + [(64,1), (64,3), (256,1)]*3 +[2]+ [(128,1), (128,3), (512,1)]*4 +[2]+\
    [(256,1), (256,3), (1024,1)]*6 +[2]+ [(512,1), (512,3), (2048,1)]*5 + [(1000,7), (1,1)]
model_res101=[(64,7), 2] + [(64,1), (64,3), (256,1)]*3 +[2]+ [(128,1), (128,3), (512,1)]*4 +[2]+\
    [(256,1), (256,3), (1024,1)]*23 +[2]+ [(512,1), (512,3), (2048,1)]*5 + [(1000,7), (1,1)]
model_res152=[(64,7), 2] + [(64,1), (64,3), (256,1)]*3 +[2]+ [(128,1), (128,3), (512,1)]*8 +[2]+\
    [(256,1), (256,3), (1024,1)]*36 +[2]+ [(512,1), (512,3), (2048,1)]*5 + [(1000,7), (1,1)]

# %% supporting functions

def parse_model(model_str):
    is_str, model_str = model_str.split(', ')
    inshape = tuple(int(v) for v in is_str.split('x'))
    res = []
    for lyr in model_str.split('-'):
        if m := re.match('(\d+)c(\d+)', lyr):
            res.append((int(m[1], int(m[2]))))
        elif m := re.match('p(\d+)', lyr):
            res.append(int(m[1]))
    return inshape, res

def comp_shape(model, inshape=(3, 28, 28)):
    # inshape=(c, h, w)
    c, h, w = inshape
    res = [(c, h, w)]
    for m in model:
        if isinstance(m, tuple): # convolution
            n, k = m
            h, w = h - k + 1, w - k + 1
        else: # pooling
            k = m
            n = c
            h, w = h//k, w//k
        res.append((n, h, w))
        c = n
    return res

def comp_shape_reverse(model, c0=3):
    c, h, w = 1, 1, 1
    res=[(c, h, w)]
    for i in range(len(model)-1, -1, -1):
        k = model[i][1]
        c = model[i-1][0] if i>0 else c0
        h, w = h + k - 1, w + k - 1
        res.append((c, h, w))
    res.reverse()
    return res

def comp_radius(model):
    r = 1
    res = [r]
    for m in reversed(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            r += k-1
        else: # pooling
            k = m
            r *= k
        res.append(r)
    res = list(reversed(res))
    return res

def comp_param(model, c=3):
    res = []
    for i, m in enumerate(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            v = n * c * k**2
        else: # pooling
            k = m
            v = 0
        res.append(v)
        c = n
    return sum(res), res

# %% compute number of queries

def attack_adaptive(model, c0=3, lmbda=1e9):
    c = c0
    Ds = comp_radius(model)
    D0, Ds = Ds[0], Ds[1:]
    res = []
    for i, m in enumerate(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            Nw = c * k**2 + 1
            Nm = 2 * n * Ds[i]**2
            # Ns = 4*Ds[i]**2 + Ds[i]**2*np.log2(lmbda)
            Ns = 4*Ds[i]**2 + n*np.log2(lmbda)
            v = Nw + Nm + Ns
        else: # pooling
            k = m
            v = 0
        res.append(v)
        c = n
    return sum(res), res

def attack_adaptive_lb(model, c0=3):
    c = c0
    Ds = comp_radius(model)
    D0, Ds = Ds[0], Ds[1:]
    res = []
    for i, m in enumerate(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            Nw = n * c * k**2
            Nm = n * Ds[i]**2
            v = Nw + Nm
        else: # pooling
            k = m
            v = 0
        res.append(v)
        c = n
    # return sum(res), res
    return max(res), res

def attack_adaptive_semi(model, c0=3):
    CF = math.log(2)
    c = c0
    Ds = comp_radius(model)
    D0, Ds = Ds[0], Ds[1:]
    res = []
    for i, m in enumerate(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            Nw = n * c * k**2
            Nm = Ds[i]**2
            Ns = math.lgamma(n + 1) / CF * Nm
            v = Nw + Nm + Ns
        else: # pooling
            k = m
            v = 0
        res.append(v)
        c = n
    return sum(res), res

def attack_adaptive_mal(model, c0=3):
    CF = math.log(2)
    c = c0
    Ds = comp_radius(model)
    D0, Ds = Ds[0], Ds[1:]
    res = []
    for i, m in enumerate(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            Nw = n * c * k**2
            Nm = k**2
            Ns = math.lgamma(n + 1) / CF * Nm
            v = Nw + Nm + Ns
        else: # pooling
            k = m
            v = 0
        res.append(v)
        c = n
    return sum(res), res

def attack_blackbox_worst(model, c0=3, lmbda=1e8):
    c = c0
    Cs = np.array([(m[0] if isinstance(m, tuple) else 0) for m in model])
    Ds = comp_radius(model)
    D0, Ds = Ds[0], Ds[1:]
    As = np.array(Ds)**2
    Bs = Cs*As
    Bs = np.cumsum(Bs[::-1])[::-1]
    res = []
    for i, m in enumerate(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            p = As[i] / Bs[i]
            En = (Cs[i]+1)/Cs[i]/p
            Np = np.log2(lmbda) + (En-1)*np.log2(lmbda/En)
            Nc = En
            #Nf = (Cs[i]-1)*p*En+1 # average value
            Nf = n # approxmated value
            # v = 2*(Np*(As[i]/k**2) + Nc*(k+1) + Nf*Cs[i-1]*k**2)
            v = 3*Np + 2*Nc*(As[i]/k**2) + 2*Nf*Cs[i-1]*k**2
            # v = 2*(np.log2(lmbda*p)/p*(L-i)**2 + Nc*(k+1) + Nf*Cs[i-1]*k**2)
        else: # pooling
            k = m
            v = 0
        res.append(v)
        c = n
    return sum(res), res

def attack_blackbox_reuse(model, c0=3, lmbda=1e8, ratio_left=0.4):
    c = c0
    Cs = np.array([(m[0] if isinstance(m, tuple) else 0) for m in model])
    Ds = comp_radius(model)
    D0, Ds = Ds[0], Ds[1:]
    As = np.array(Ds)**2
    Bs = Cs*As
    Bs = np.cumsum(Bs[::-1])[::-1]
    res = []
    for i, m in enumerate(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            p = As[i] / Bs[i]
            En = (Cs[i]+1)/Cs[i]/p
            En_new = (ratio_left*En) if i != 0 else En
            Np = np.log2(lmbda) + (En_new-1)*np.log2(lmbda/En)
            Nc = En
            Nf = n
            # v = 2*(Np + Nc*(k+1) + Nf*Cs[i-1]*k**2)
            v = 3*Np + 2*Nc*(As[i]/k**2) + 2*Nf*Cs[i-1]*k**2
        else: # pooling
            k = m
            v = 0
        res.append(v)
        c = n
    return sum(res), res

def attack_exp(model, c0=3):
    c = c0
    Cs = np.array([(m[0] if isinstance(m, tuple) else 0) for m in model])
    C = sum(Cs)
    Ds = comp_radius(model)
    D0, Ds = Ds[0], Ds[1:]
    res = []
    for i, m in enumerate(model):
        if isinstance(m, tuple): # convolution
            n, k = m
            lmd = Ds[i]**2
            # print(C, lmd, C*lmd, np.log2(C*lmd))
            v = C * np.log(n) * (np.log2(C)+np.log2(lmd)) * c * (Ds[i]**2*c/2/k/k + k**2)
            C -= n
        else: # pooling
            k = m
            v = 0
        res.append(v)
        # prepare for next layer
        c = n
    return sum(res), res

# %%

def show(model, c0=3, lmbda=1e9):
    param = comp_param(model, c0)[0]
    data = (
        attack_adaptive(model, c0, lmbda)[0],
        # attack_adaptive_lb(model, c0)[0],
        # attack_adaptive_mal(model, c0)[0],
        attack_blackbox_worst(model, c0, lmbda)[0],
        attack_blackbox_reuse(model, c0, lmbda, 0.4)[0],
        attack_exp(model, c0)[0]
        )
    logdata = np.log2(data)
    print(param, ' '.join('%.4f'%v for v in logdata))

# %% main

def test():
    c0=3
    lmbda=1e12
    # lmbda=1e16

    show(model1, c0, lmbda)
    show(model2, c0, lmbda)
    show(model3, c0, lmbda)
    show(model4, c0, lmbda)
    show(model5, c0, lmbda)
    show(model6, c0, lmbda)

    show(model_lenet, c0, lmbda)
    show(model_alex, c0, lmbda)

    # lmbda=1e8
    show(model_vgg11, c0, lmbda)
    show(model_vgg13, c0, lmbda)
    show(model_vgg16, c0, lmbda)
    show(model_vgg19, c0, lmbda)

    show(model_res18, c0, lmbda)
    show(model_res34, c0, lmbda)
    show(model_res50, c0, lmbda)
    show(model_res101, c0, lmbda)
    show(model_res152, c0, lmbda)

