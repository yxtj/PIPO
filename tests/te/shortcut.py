import torch_extension as te

import torch
import torch.nn as nn

# %% test shortcut layer

def basic_test():
    layer_a = te.Addition(-2)
    layer_c = te.Concatenation(-2)
    a = torch.ones(1, 1, 3, 3)+0.1
    b = torch.ones(1, 1, 3, 3)+0.2
    layer_a.update(-2, a)
    layer_c.update(-2, a)
    ya = layer_a(b)
    yc = layer_c(b)
    print(ya, a+b)
    print(yc, torch.cat((a, b), dim=1))

basic_test()

# %% test sequential shortcut

def basic_seqsc_test():
    inshape = (1, 3)
    model = te.SequentialShortcut(
        nn.Linear(3, 4),
        nn.Linear(4, 4),
        te.Addition(-2),
    )
    x = torch.ones(inshape)
    buffer = [x]
    with torch.no_grad():
        # layer 0 (linear 3-4)
        d = model[0](x)
        buffer.append(d)
        # layer 1 (linear 4-4)
        d = model[1](d)
        buffer.append(d)
        # layer 2 (add)
        d = buffer[1] + buffer[2]
        buffer.append(d)
        print(buffer)
        y = model(x)
        print(y)
        print((y-buffer[-1]).pow(2).sum().sqrt())

basic_seqsc_test()

# %% case tests

def model_test(model, inshape, show=False):
    x = torch.ones(inshape)
    with torch.no_grad():
        y = model(x)
    if show:
        print(y)

    d = x
    buff = [d]
    for i, lyr in enumerate(model):
        if show:
            print(i, lyr)
        if isinstance(lyr, te.ShortCut):
            t = {}
            for j in lyr.buffer.keys(): # use buffer.keys() to get the specified order
                t[j] = buff[i + j + 1]
            lyr.buffer = t
        d = lyr(d)
        # print(d)
        buff.append(d)
    if show:
        print(d)

    print("diff:", (y-d).pow(2).sum().sqrt())

# %% test add with conv

print("Add with input, conv")
model_test(te.SequentialShortcut(
    nn.Conv2d(2, 2, 3, 1, 1),
    te.Addition(-2),
), (1, 2, 5, 5))

print("Add with intermediate result, conv")
model_test(te.SequentialShortcut(
    nn.Conv2d(1, 2, 3, 1, 1),
    nn.Conv2d(2, 2, 3, 1, 1),
    te.Addition(-2),
), (1, 1, 5, 5))

print("Add with multiple intermediate results, conv")
model_test(te.SequentialShortcut(
    nn.Conv2d(1, 2, 3, 1, 1),
    nn.Conv2d(2, 2, 3, 1, 1),
    nn.Conv2d(2, 2, 3, 1, 1),
    te.Addition([-2, -3]),
), (1, 1, 5, 5))

# %% test add with fc

print("Add with input, fc")
model_test(te.SequentialShortcut(
    nn.Linear(10, 10),
    te.Addition(-2),
), (1, 10))

print("Add with intermediate result, fc")
model_test(te.SequentialShortcut(
    nn.Linear(10, 10),
    nn.Linear(10, 10),
    te.Addition(-2),
), (1, 10))

# %% test concat with fc

print("Concat with input, fc")
model_test(te.SequentialShortcut(
    nn.Linear(10, 10),
    te.Concatenation(-2),
), (1, 10))

print("Concat with intermediate result, fc")
model_test(te.SequentialShortcut(
    nn.Linear(10, 10),
    nn.Linear(10, 20),
    te.Concatenation(-2),
), (1, 10))

print("Concat with multiple intermediate results, fc")
model_test(te.SequentialShortcut(
    nn.Linear(10, 10),
    nn.Linear(10, 20),
    nn.Linear(20, 30),
    te.Concatenation([-2, -3]),
), (1, 10))

print("Concat with multiple intermediate results (non-trival order), fc")
model_test(te.SequentialShortcut(
    nn.Linear(10, 10),
    nn.Linear(10, 20),
    nn.Linear(20, 30),
    te.Concatenation([-2, -3], 1, [-3, -1, -2]),
), (1, 10))
