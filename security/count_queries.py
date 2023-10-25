# -*- coding: utf-8 -*-
import numpy as np
import math

def f(n, c, k):
    return n*c*k*k

model1=[(10,3), (1,3)]
model2=[(32,3), (16,3), (1,3)]
model3=[(16,5), (16,5), (1,5)]
model4=[(10,3), (10,3), (10,3), (1,3)]
model5=[(10,3), (10,3), (10,3), (10,3), (1,3)]
model6=[(6,5), (6,5), (16,5), (4,5), (1,12)] # mnist

def comp_shape(model, inshape=(3, 28, 28)):
    # inshape=(c, h, w)
    c, h, w = inshape
    res = [(c, h, w)]
    for n, k in model:
        h, w = h - k + 1, w - k + 1
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


def comp_param(model, c=3):
    res = []
    for n, k in model:
        v = f(n, c, k)
        res.append(v)
        c = n
    return sum(res), res

def comp_adaptive_lb(model, c0=3):
    CF = math.log(2)
    c = c0
    L = len(model)
    Ks = np.array([k for n,k in model])
    Ds = [Ks[i]-1 for i in range(L)] + [1]
    Ds = np.cumsum(Ds[::-1])
    D0, Ds = Ds[-1], Ds[-2::-1]
    res = []
    for i, (n, k) in enumerate(model):
        Nw = n * c * k**2
        Nm = n * Ds[i]**2
        v = Nw + Nm
        res.append(v)
        c = n
    return sum(res), res

def comp_adaptive_semi(model, c0=3):
    CF = math.log(2)
    c = c0
    L = len(model)
    Ks = np.array([k for n,k in model])
    Ds = [Ks[i]-1 for i in range(L)] + [1]
    Ds = np.cumsum(Ds[::-1])
    D0, Ds = Ds[-1], Ds[-2::-1]
    res = []
    for i, (n, k) in enumerate(model):
        Nw = n * c * k**2
        Nm = Ds[i]**2
        Ns = math.lgamma(n + 1) / CF * Nm
        v = Nw + Nm + Ns
        res.append(v)
        c = n
    return sum(res), res

def comp_adaptive_mal(model, c0=3):
    CF = math.log(2)
    c = c0
    L = len(model)
    Ks = np.array([k for n,k in model])
    Ds = [Ks[i]-1 for i in range(L)] + [1]
    Ds = np.cumsum(Ds[::-1])
    D0, Ds = Ds[-1], Ds[-2::-1]
    res = []
    for i, (n, k) in enumerate(model):
        Nw = n * c * k**2
        Nm = k**2
        Ns = math.lgamma(n + 1) / CF * Nm
        v = Nw + Nm + Ns
        res.append(v)
        c = n
    return sum(res), res

def comp_oracle(model, c0=3, lmbda=1e8):
    c = c0
    L = len(model)
    Cs = np.array([n for n,k in model])
    Ks = np.array([k for n,k in model])
    Ds = [Ks[i]-1 for i in range(L)] + [1]
    Ds = np.cumsum(Ds[::-1])
    D0, Ds = Ds[-1], Ds[-2::-1]
    As = Ds**2
    Bs = Cs*As
    Bs = np.cumsum(Bs[::-1])[::-1]
    res = []
    for i, (n, k) in enumerate(model):
        p = As[i] / Bs[i]
        En = (Cs[i]+1)/Cs[i]/p
        Np = np.log2(lmbda) + (En-1)*np.log2(lmbda/En)
        Nc = En
        #Nf = (Cs[i]-1)*p*En+1 # average value
        Nf = n # minimum value
        v = 2*(Np*(As[i]/k**2) + Nc*(k+1) + Nf*Cs[i-1]*k**2)
        # v = 2*(np.log2(lmbda*p)/p*(L-i)**2 + Nc*(k+1) + Nf*Cs[i-1]*k**2)
        res.append(v)
        # prepare for next layer
        c = n
    return sum(res), res

def comp_exp(model, c0=3):
    c = c0
    L = len(model)
    C = sum(n for n,k in model)
    K = 1+sum(k-1 for n,k in model)
    K0 = K**2*c
    res = []
    for i, (n, k) in enumerate(model):
        lmd = K**2
        v = C * np.log(n) * np.log2(C*lmd) * c * (K0/2/k/k + k**2)
        res.append(v)
        # prepare for next layer
        C -= n
        K -= k-1
        c = n
    return sum(res), res

def show(model, c0=3, lmbda=1e9):
    param = comp_param(model, c0)[0]
    data = (
        comp_adaptive_lb(model, c0)[0],
        comp_adaptive_mal(model, c0)[0],
        comp_oracle(model, c0, lmbda)[0],
        comp_exp(model, c0)[0]
        )
    logdata = np.log2(data)
    print(param, ' '.join('%.4f'%v for v in logdata))

# %% main

def test():
    c0=3
    lmbda=1e12
    show(model1, c0, lmbda)
    show(model2, c0, lmbda)
    show(model3, c0, lmbda)
    show(model4, c0, lmbda)
    show(model5, c0, lmbda)
    show(model6, c0, lmbda)
