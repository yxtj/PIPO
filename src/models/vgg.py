# reference: https://github.com/SeHwanJoo/cifar10-vgg16/blob/master/vgg16.py

# import torch
import torch.nn as nn

inshape = (3, 224, 224)


def make_block(inch, outch, n):
    l = [nn.Conv2d(inch, outch, 3, 1, 1)]
    l += [nn.Conv2d(outch, outch, 3, 1, 1) for i in range(n-1)]
    l += [nn.MaxPool2d(2, 2)]
    return l


def build_vgg(layers=16, fc=True):
    assert layers in [11, 13, 16, 19]
    # 3*224*224 -conv*-> 64*224*224 -max-> 128*112*112
    # -conv-> 256*112*112 -conv-> 256*112*112 -max-> 256*56*56
    # -conv-> 512*56*56 -conv*-> 512*56*56 -max-> 512*28*28
    # -conv-> 512*28*28 -conv*-> 512*28*28 -max-> 512*14*14
    # -conv-> 512*14*14 -conv*-> 512*14*14 -max-> 512*7*7
    # -fc-> 4096 -fc-> 4096 -fc-> 1000
    if layers == 11:
        l1 = [*make_block(3, 64, 1), *make_block(64, 128, 1),
                *make_block(128, 256, 2), *make_block(256, 512, 2),
                *make_block(512, 512, 2)]
    elif layers == 13:
        l1 = [*make_block(3, 64, 2), *make_block(64, 128, 2),
                *make_block(128, 256, 2), *make_block(256, 512, 2),
                *make_block(512, 512, 2)]
    elif layers == 16:
        l1 = [*make_block(3, 64, 2), *make_block(64, 128, 2),
                *make_block(128, 256, 3), *make_block(256, 512, 3),
                *make_block(512, 512, 3)]
    elif layers == 19:
        l1 = [*make_block(3, 64, 2), *make_block(64, 128, 2),
                *make_block(128, 256, 4), *make_block(256, 512, 4),
                *make_block(512, 512, 4)]
    if fc:
        l2 = [
            nn.Linear(7*7*512, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 1000)
            ]
        model = nn.Sequential(*l1, nn.Flatten(), *l2) #, nn.Softmax(dim=1))
    else:
        model = nn.Sequential(*l1) #, nn.Softmax(dim=1))
    return model


def build_vgg11():
    return build_vgg(11)

def build_vgg13():
    return build_vgg(13)

def build_vgg16():
    return build_vgg(16)

def build_vgg19():
    return build_vgg(19)
