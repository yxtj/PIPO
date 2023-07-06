import torch_extension as te

import torch
import torch.nn as nn


# test shortcut layer

layer = te.ShortCut(-2)
a = torch.ones(1, 1, 3, 3)+0.1
b = torch.ones(1, 1, 3, 3)+0.2
y = layer(a, b)
print(y)

# test sequential buffer
    
def model_test(model, inshape):
    x = torch.ones(inshape)
    with torch.no_grad():
        y = model(x)
    print(y)

    d = x
    buff = []
    for i, lyr in enumerate(model):
        print(i, lyr)
        if isinstance(lyr, te.ShortCut):
            d = lyr(d, buff[i + lyr.otherlayer])
        else:
            d = lyr(d)
        # print(d)
        buff.append(d)
    print(d)

    print("diff:", (y-d).pow(2).sum().sqrt())

# test 1

inshape1 = (1, 1, 5, 5)
model1 = te.SequentialBuffer(
    nn.Conv2d(1, 2, 3, 1, 1),
    nn.Conv2d(2, 2, 3, 1, 1),
    te.ShortCut(-2),
)

model_test(model1, inshape1)

# test 2

inshape2 = (1, 10)
model2 = te.SequentialBuffer(
    nn.Linear(10, 10),
    nn.Linear(10, 10),
    te.ShortCut(-2),
)

model_test(model2, inshape2)

# test 3

inshape3 = (1, 1, 5, 5)
model3 = te.SequentialBuffer(
    nn.Conv2d(1, 2, 3, 1, 1),
    te.ShortCut(-1),
)

model_test(model3, inshape3)

