# -*- coding: utf-8 -*-
import torch

import src.train.util
import src.poc.noisy_model

import src.models.minionn
import src.models.resnet


# %% minionn

_,ds_c10=ml.util.load_data('cifar10', 'E:/Data/CIFAR10/',False,True)

m = model.minionn.build()
ml.util.load_model_state(m, 'pretrained/minionn.pt')

# no noise
ml.util.test(m, ds_c10, 100, device='cuda')

# 0.9034

noise_list = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]

# with noise
acc_list=[]
for r in noise_list:
    m2=poc.noisy_model.NoisyModel(m, r)
    a=ml.util.test(m2, ds_c10, 100, device='cuda')
    print(r,'  ',a)
    acc_list.append(a)

# [0.9037, 0.9034, 0.901, 0.8944, 0.8788, 0.8596, 0.8282]

# %% resnet

_,ds_c100=ml.util.load_data('cifar100', 'E:/Data/CIFAR100/',False,True)

m = model.resnet.build(32, 3)
ml.util.load_model_state(m, 'pretrained/resnet-3.pt')

# no noise
ml.util.test(m, ds_c100, 100, device='cuda')

# 0.7642

noise_list = [0.01,0.05,0.1,0.2,0.3,0.4,0.5]

# with noise
acc_list=[]
for r in noise_list:
    m2=poc.noisy_model.NoisyModel_ext(m, r)
    a=ml.util.test(m2, ds_c100, 100, device='cuda')
    print(r,'  ',a)
    acc_list.append(a)

# [0.7642, 0.7627, 0.7597, 0.7315, 0.6644, 0.5499, 0.4235]
