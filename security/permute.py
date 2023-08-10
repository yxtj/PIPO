# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch_extension.shortcut as te
import numpy as np


def comp_out_shape(lyr, inshape, idx=None, shapes=None):
    if isinstance(lyr, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
        pad = (lyr.padding, lyr.padding) if isinstance(lyr.padding, int) else lyr.padding
        ks = (lyr.kernel_size, lyr.kernel_size) if isinstance(lyr.kernel_size, int) else lyr.kernel_size
        std = (lyr.stride, lyr.stride) if isinstance(lyr.stride, int) else lyr.stride
        h = (inshape[1] + 2*pad[0] - ks[0]) / std[0] + 1
        w = (inshape[2] + 2*pad[1] - ks[1]) / std[1] + 1
        if isinstance(lyr, nn.Conv2d) or lyr.ceil_mode==False:
            h = int(h)
            w = int(w)
        else:
            h = int(np.ceil(h))
            w = int(np.ceil(w))
        c = lyr.out_channels if isinstance(lyr, nn.Conv2d) else inshape[0]
        outshape = (c, h, w)
    elif isinstance(lyr, nn.Flatten):
        of = np.prod(inshape[1:])
        outshape = (of, )
    elif isinstance(lyr, nn.Linear):
        outshape = (lyr.out_features, )
    elif isinstance(lyr, te.Jump):
        outshape = shapes[idx+lyr.relOther[0]+1]
    elif isinstance(lyr, te.Concatenation):
        of = sum(shapes[idx+i+1][lyr.dim-1] for i in lyr.relOther)
        o = [*inshape]
        o[lyr.dim-1] += of
        outshape = tuple(o)
    else:
        outshape = inshape
    return outshape

# get the lower bound of breaking a premutation of a convolution layer
def break_conv_permute_one(lyr:nn.Conv2d, inshape:tuple) -> int:
    # n = lyr.out_channels
    # c = lyr.in_channels
    # h, w = lyr.kernel_size
    # kernel parameters
    n, c, h, w = lyr.weight.shape
    k = c*h*w
    # data parameters
    C, H, W = inshape
    # coverage
    cvg_c = c*2 - 1
    cvg_h = h*2 - 1
    cvg_w = w*2 - 1
    cvg = cvg_c * cvg_h * cvg_w
    # bit of information
    # res = int(np.math.lgamma(C*H*W+1) / np.log(2+1)) * n
    # count
    # res = cvg * int(np.ceil(np.log2(C*H*W / (cvg*2-1)))) * n
    res = int(np.ceil(np.log2(2*cvg - 1)) * np.ceil(np.log2(C*H*W / (2*cvg-1)))) * n
    # res = int(np.ceil(np.log2(C*H*W / cvg))) * n
    # res = int(np.ceil(np.log2(C*H*W / (cvg*2-1)))) * n
    return res


def break_conv_permute_model(m, inshape):
    counts = []
    shapes = []
    s = inshape
    shapes.append(s)
    for i, lyr in enumerate(m):
        if isinstance(lyr, nn.Conv2d):
            c = break_conv_permute_one(lyr, s)
            counts.append(c)
        s = comp_out_shape(lyr, s, i, shapes)
        # print(i, lyr.__class__.__name__, shapes[i], s, c if isinstance(lyr, nn.Conv2d) else None)
        shapes.append(s)
    return counts #, shapes

