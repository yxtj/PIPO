import torch_extension as te


import torch

import torch.nn as nn



# test shortcut layer


layer_a = te.Addition(-2)

layer_c = te.Concatenation(-2)

a = torch.ones(1, 1, 3, 3)+0.1

b = torch.ones(1, 1, 3, 3)+0.2

layer_a.update(a)

layer_c.update(a)

ya = layer_a(b)

yc = layer_c(b)

print(ya, layer_a.forward2(a, b))

print(yc, layer_c.forward2(a, b))



# test sequential buffer


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

    d = model[2].forward2(buffer[1], d)

    buffer.append(d)

    print(buffer)

    y = model(x)
    print(y)

    print((y-buffer[-1]).pow(2).sum().sqrt())


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

            d = lyr.forward2(buff[i + lyr.relOther + 1], d)

        else:

            d = lyr(d)

        # print(d)

        buff.append(d)

    if show:
        print(d)


    print("diff:", (y-d).pow(2).sum().sqrt())


# test 1


print("Add with input, conv")

model_test(te.SequentialShortcut(

    nn.Conv2d(1, 2, 3, 1, 1),

    te.Addition(-2),

), (1, 1, 5, 5))


# test 2


print("Add with intermediate result, conv")

model_test(te.SequentialShortcut(

    nn.Conv2d(1, 2, 3, 1, 1),

    nn.Conv2d(2, 2, 3, 1, 1),

    te.Addition(-2),

), (1, 1, 5, 5))


# test 3


print("Add with input, fc")

model_test(te.SequentialShortcut(

    nn.Linear(10, 10),

    te.Addition(-2),

), (1, 10))


# test 4


print("Add with intermediate result, fc")

model_test(te.SequentialShortcut(

    nn.Linear(10, 10),

    nn.Linear(10, 10),

    te.Addition(-2),

), (1, 10))


# test 5


print("Concat with input, fc")

model_test(te.SequentialShortcut(

    nn.Linear(10, 10),

    te.Concatenation(-2),

), (1, 10))


# test 6


print("Concat with intermediate result, fc")

model_test(te.SequentialShortcut(

    nn.Linear(10, 10),

    nn.Linear(10, 20),

    te.Concatenation(-2),

), (1, 10))

