# import torch
import torch.nn as nn
import torch_extension as te

map = {}

# Model 0 linear:
# Shape: 6 -> 2

Poc0Inshape_l = (6,)
Poc0Model_l = nn.Sequential(
    nn.Linear(6, 2),
)
map["0-linear"] = (Poc0Inshape_l, Poc0Model_l)

Poc0Model_l2 = nn.Sequential(
    nn.Linear(6, 3),
    nn.Linear(3, 2),
)
map["0-l2"] = (Poc0Inshape_l, Poc0Model_l2)

Poc0Model_l3 = nn.Sequential(
    nn.Linear(6, 3),
    nn.ReLU(),
    nn.Linear(3, 2),
)
map["0-l3"] = (Poc0Inshape_l, Poc0Model_l3)

Poc0Model_l4 = nn.Sequential(
    nn.Linear(6, 5),
    nn.Linear(5, 4),
    nn.Linear(4, 3),
    nn.Linear(3, 2),
)
map["0-l4"] = (Poc0Inshape_l, Poc0Model_l4)

# Model 0 pool:
# Shape: 1x6x6 -> 1x2x2 -> 4 -> 2

Poc0Inshape_p = (1, 6, 6)
Poc0Model_m = nn.Sequential(
    nn.MaxPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(4, 2),
)
Poc0Model_a = nn.Sequential(
    nn.AvgPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(4, 2),
)
map["0-max"] = (Poc0Inshape_p, Poc0Model_m)
map["0-avg"] = (Poc0Inshape_p, Poc0Model_a)

# Model 0 shortcut:
# Shape: 2 -> 2

Poc0Inshape_s = (2,)
Poc0Model_s = te.SequentialBuffer(
# nn.Sequential(
    nn.Linear(2, 2),
    nn.Linear(2, 2),
    te.ShortCut(-2)
)
map["0-sc"] = (Poc0Inshape_s, Poc0Model_s)

# Model 1:
# Shape: 1x10x10 -> 5x8x8  -> 10x6x6 -> 360 -> 10

Poc1Inshape = (1, 10, 10)
Poc1Model = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 10, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(360, 10),
    # nn.Softmax()
)
map["1"] = (Poc1Inshape, Poc1Model)

# Model 2:
# Shape: 1x10x10 -> 5x8x8 -> 5x2x2 -> 20 -> 10

Poc2Inshape = (1, 10, 10)
Poc2Model_max = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.MaxPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(20, 10),
)
map["2"] = (Poc2Inshape, Poc2Model_max)

Poc2Model_avg = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.AvgPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(20, 10),
)
map["2-avg"] = (Poc2Inshape, Poc2Model_avg)

# Model 3:
# Shape: 1x32x32 -conv-> 5x30x30 -max-> 5x10x10 -conv-> 10x8x8 -max-> 10x2x2 -> 40 -> 10

Poc3Inshape = (1, 32, 32)
Poc3Model = nn.Sequential(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.MaxPool2d(3, 3),
    nn.Conv2d(5, 10, 3),
    nn.ReLU(),
    nn.MaxPool2d(3, 3),
    nn.Flatten(),
    nn.Linear(40, 10),
)
map["3"] = (Poc3Inshape, Poc3Model)

# Model 4:
# Shape: 10 -relu-> 10 -shortcut(-1,-3)-> 10 -> 5

Poc4Inshape = (10,)
Poc4Model = te.SequentialBuffer(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    te.ShortCut(-2),
    nn.ReLU(),
    nn.Linear(10, 5),
)
map["4"] = (Poc4Inshape, Poc4Model)

# Model 5:
# Shape: 1x10x10 -conv-> 5x8x8 -conv-> 5x8x8 -shortcut(-3)-> 5x8x8 -> 320 -> 10

Poc5Inshape = (1, 10, 10)
Poc5Model = te.SequentialBuffer(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 5, 3, 1, 1),
    nn.ReLU(),
    nn.Conv2d(5, 5, 3, 1, 1),
    te.ShortCut(-2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(320, 10),
)
map["5"] = (Poc5Inshape, Poc5Model)
