# reference: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

# import torch
import torch.nn as nn
import torch_extension as te

inshape = (3, 32, 32)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, 
                     stride=stride, padding=1, bias=False)

def build_downsample_block(in_channels, out_channels, stride, batch_norm):
    if batch_norm:
        layers = [
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        ]
        if stride == 1:
            layers.append(te.ShortCut(-6))
    else:
        layers = [
            conv3x3(in_channels, out_channels, stride),
            nn.ReLU(),
            conv3x3(out_channels, out_channels),
        ]
        if stride == 1:
            layers.append(te.ShortCut(-4))
    layers.append(nn.ReLU())
    return layers
    
def build_identity_block(channels, batch_norm):
    if batch_norm:
        layers = [
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            conv3x3(channels, channels),
            nn.BatchNorm2d(channels),
            te.ShortCut(-6),
            nn.ReLU(),
        ]
    else:
        layers = [
            conv3x3(channels, channels),
            nn.ReLU(),
            conv3x3(channels, channels),
            te.ShortCut(-4),
            nn.ReLU(),
        ]
    return layers

def build_block(layer_size, in_channels, out_channels, stride, batch_norm):
    layers = build_downsample_block(in_channels, out_channels, stride, batch_norm)
    for i in range(layer_size-1):
        layers.extend(build_identity_block(out_channels, batch_norm))
    return layers

def build_resnet(num_blocks, num_class=100, version=1, residual=True, batch_norm=False):
    # layer 0: 3x32x32 -> 16x32x32
    layers = [ conv3x3(3, 16), nn.ReLU() ]
    # layer 1: 16x32x32 -> 16x32x32
    layers.extend(build_block(num_blocks[0], 16, 16, 1, batch_norm))
    # layer 2: 16x32x32 -> 32x16x16
    layers.extend(build_block(num_blocks[1], 16, 32, 2, batch_norm))
    # layer 3: 32x16x16 -> 64x8x8
    layers.extend(build_block(num_blocks[2], 32, 64, 2, batch_norm))
    # pooling and fc
    if version == 1: # cifar10
        layers.extend([
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64, num_class),
        ])
    elif version == 2: # cifar10
        layers.extend([
            nn.Conv2d(64, 64, 8, 8, bias=False),
            nn.Flatten(),
            nn.Linear(64, num_class),
        ])
    elif version == 3: # cifar100
        layers.extend([
            nn.Flatten(),
            nn.Linear(4096, num_class),
        ])
    elif version == 4: # cifar100
        layers.extend([
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, num_class),
        ])
    if not residual:
        layers = [ lyr for lyr in layers if not isinstance(lyr, te.ShortCut) ]
    #layers.append(nn.Softmax(dim=1))
    return te.SequentialBuffer(*layers)


def resnet20(version=3, residual=True, batch_norm=False):
    return build_resnet([3, 3, 3], 100, version, residual, batch_norm)

def resnet32(version=3, residual=True, batch_norm=False):
    return build_resnet([5, 5, 5], 100, version, residual, batch_norm)

def resnet44(version=3, residual=True, batch_norm=False):
    return build_resnet([7, 7, 7], 100, version, residual, batch_norm)

def resnet56(version=3, residual=True, batch_norm=False):
    return build_resnet([9, 9, 9], 100, version, residual, batch_norm)

def resnet110(version=3, residual=True, batch_norm=False):
    return build_resnet([18, 18, 18], 100, version, residual, batch_norm)

def resnet152(version=3, residual=True, batch_norm=False):
    return build_resnet([24, 24, 24], 100, version, residual, batch_norm)


def build(depth, version=3, residual=True, batch_norm=False):
    if depth == 20:
        return resnet20(version, residual, batch_norm)
    elif depth == 32:
        return resnet32(version, residual, batch_norm)
    elif depth == 44:
        return resnet44(version, residual, batch_norm)
    elif depth == 56:
        return resnet56(version, residual, batch_norm)
    elif depth == 110:
        return resnet110(version, residual, batch_norm)
    elif depth == 152:
        return resnet152(version, residual, batch_norm)
    else:
        raise ValueError("depth must be 20, 32, 44, 56, 110, or 152")
