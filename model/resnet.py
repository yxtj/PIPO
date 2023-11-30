# import torch
import torch.nn as nn
import torch_extension as te

inshape = (3, 224, 224)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, 
                     stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, 
                     stride=stride, padding=0, bias=False)

def basicblock(in_channels, out_channels, stride=1, batch_norm=False):
    downsample = in_channels != out_channels or stride != 1
    norm = nn.BatchNorm2d if batch_norm else lambda x: None
    if batch_norm:
        offset_ds = -6
        offset_add = -4 if downsample else offset_ds
    else:
        offset_ds = -4
        offset_add = -3 if downsample else offset_ds
    layers = [
        conv3x3(in_channels, out_channels, stride),
        norm(out_channels),
        nn.ReLU(),
        conv3x3(out_channels, out_channels),
        norm(out_channels),
    ]
    if downsample:
        layers.extend([
            te.Jump(offset_ds),
            conv1x1(in_channels, out_channels, stride),
            norm(out_channels),
        ])
    layers = [ lyr for lyr in layers if lyr is not None ]
    layers.append(te.Addition(offset_add))
    layers.append(nn.ReLU())
    return layers

def bottleneck(in_planes, out_planes, stride=1, batch_norm=False):
    expansion = 4
    in_channels = in_planes * expansion
    mid_channels = out_planes
    out_channels = out_planes * expansion
    downsample = in_planes != out_planes or stride != 1
    norm = nn.BatchNorm2d if batch_norm else lambda x: None
    if batch_norm:
        offset_ds = -9
        offset_add = -4 if downsample else offset_ds
    else:
        offset_ds = -6
        offset_add = -3 if downsample else offset_ds
    layers = [
        conv1x1(in_channels, mid_channels, stride),
        norm(mid_channels),
        nn.ReLU(),
        conv3x3(mid_channels, mid_channels),
        norm(mid_channels),
        nn.ReLU(),
        conv1x1(mid_channels, out_channels),
        norm(out_channels),
    ]
    if downsample:
        layers.extend([
            te.Jump(offset_ds),
            conv1x1(in_channels, out_channels, stride),
            norm(out_channels),
        ])
    layers = [ lyr for lyr in layers if lyr is not None ]
    layers.append(te.Addition(offset_add))
    layers.append(nn.ReLU())
    return layers

def build_resnet(block, num_blocks, num_class=1000, batch_norm=False):
    expansion = 1 if block == basicblock else 4
    # conv1: 3x224x224 -> 64x112x112
    if batch_norm:
        layers = [ nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU()]
    else:
        layers = [ nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.ReLU(), ]
    # maxpool: 64x112x112 -> 64x56x56
    # layers.append(nn.MaxPool2d(3, 2, 1))
    layers.append(nn.MaxPool2d(2, 2, 0)) # the kernel size should be 3x3, will support in the future
    # layer 1: 64x56x56 -> 64x56x56, or 64x56x56 -> 256x56x56
    layers.extend(block(64//expansion, 64, batch_norm=batch_norm))
    # layer 2: 64x56x56 -> 128x28x28, or 256x56x56 -> 512x28x28
    layers.extend(block(64, 128, 2, batch_norm=batch_norm))
    # layer 3: 128x28x28 -> 256x14x14, or 512x28x28 -> 1024x14x14
    layers.extend(block(128, 256, 2, batch_norm=batch_norm))
    # layer 4: 256x14x14 -> 512x7x7, or 1024x14x14 -> 2048x7x7
    layers.extend(block(256, 512, 2, batch_norm=batch_norm))
    # pooling and fc
    layers.extend([
        nn.AvgPool2d(7),
        nn.Flatten(),
        nn.Linear(512*expansion, num_class),
    ])
    return te.SequentialShortcut(*layers)

def build(depth, num_class=1000, batch_norm=False):
    if depth == 18:
        return build_resnet(basicblock, [2, 2, 2, 2], num_class, batch_norm)
    elif depth == 34:
        return build_resnet(basicblock, [3, 4, 6, 3], num_class, batch_norm)
    elif depth == 50:
        return build_resnet(bottleneck, [3, 4, 6, 3], num_class, batch_norm)
    elif depth == 101:
        return build_resnet(bottleneck, [3, 4, 23, 3], num_class, batch_norm)
    elif depth == 152:
        return build_resnet(bottleneck, [3, 8, 36, 3], num_class, batch_norm)
    else:
        raise ValueError("depth must be in [18, 34, 50, 101, 152]")
