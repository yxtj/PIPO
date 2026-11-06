import torch
import torch.nn as nn

from src.model.dag_model import AddOp, ConcatOp, DagModel, JumpOp


# %% test shortcut layers

def basic_ops_test():
    a = torch.ones(1, 1, 3, 3) + 0.1
    b = torch.ones(1, 1, 3, 3) + 0.2
    ya = AddOp()(a, b)
    yc = ConcatOp(1)(a, b)
    print(ya, a + b)
    print(yc, torch.cat((a, b), dim=1))

basic_ops_test()

# %% test DagModel basics

def basic_dag_test():
    inshape = (1, 3)
    model = DagModel(
        nn.Linear(3, 4),
        nn.Linear(4, 4),
        (AddOp(), [-1]),
    )
    x = torch.ones(inshape)
    with torch.no_grad():
        d0 = model[0](x)
        d1 = model[1](d0)
        d2 = d0 + d1
        y = model(x)
    print("diff:", (y - d2).pow(2).sum().sqrt())

basic_dag_test()

# %% plain-forward reference test

def model_test(model, inshape, show=False):
    x = torch.ones(inshape)
    with torch.no_grad():
        y = model(x)
    if show:
        print(y)

    buff = [x]
    entries = [e for e in model.flat_entries() if not e.is_block]
    fm = {0: x}
    for i, e in enumerate(entries):
        if show:
            print(i, e.module, e.sources)
        inputs = [fm[s] for s in e.sources]
        with torch.no_grad():
            d = e.module(*inputs) if len(inputs) > 1 else e.module(inputs[0])
        fm[e.dest] = d
    y_ref = fm[len(entries)]
    if show:
        print(y_ref)
    print("diff:", (y - y_ref).pow(2).sum().sqrt())

# %% test add with conv

print("Add with input, conv")
model_test(DagModel(
    nn.Conv2d(2, 2, 3, 1, 1),
    (AddOp(), [-1]),
), (1, 2, 5, 5))

print("Add with intermediate result, conv")
model_test(DagModel(
    nn.Conv2d(1, 2, 3, 1, 1),
    nn.Conv2d(2, 2, 3, 1, 1),
    (AddOp(), [-1]),
), (1, 1, 5, 5))

print("Add with multiple intermediate results, conv")
model_test(DagModel(
    nn.Conv2d(1, 2, 3, 1, 1),
    nn.Conv2d(2, 2, 3, 1, 1),
    nn.Conv2d(2, 2, 3, 1, 1),
    (AddOp(), [-1, -2]),
), (1, 1, 5, 5))

# %% test add with fc

print("Add with input, fc")
model_test(DagModel(
    nn.Linear(10, 10),
    (AddOp(), [-1]),
), (1, 10))

print("Add with intermediate result, fc")
model_test(DagModel(
    nn.Linear(10, 10),
    nn.Linear(10, 10),
    (AddOp(), [-1]),
), (1, 10))

# %% test concat with fc

print("Concat with input, fc")
model_test(DagModel(
    nn.Linear(10, 10),
    (ConcatOp(1), [-1, None]),
), (1, 10))

print("Concat with intermediate result, fc")
model_test(DagModel(
    nn.Linear(10, 10),
    nn.Linear(10, 20),
    (ConcatOp(1), [-1, None]),
), (1, 10))

print("Concat with multiple intermediate results, fc")
model_test(DagModel(
    nn.Linear(10, 10),
    nn.Linear(10, 20),
    nn.Linear(20, 30),
    (ConcatOp(1), [-1, -2, None]),
), (1, 10))

print("Concat with multiple intermediate results (non-trivial order), fc")
model_test(DagModel(
    nn.Linear(10, 10),
    nn.Linear(10, 20),
    nn.Linear(20, 30),
    (ConcatOp(1), [-2, None, -1]),
), (1, 10))

# %% test jump

print("Jump, fc")
model_test(DagModel(
    nn.Linear(10, 10),
    (JumpOp(), [-1]),
    nn.Linear(10, 10),
), (1, 10))
