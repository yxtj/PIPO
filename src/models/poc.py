import torch.nn as nn
from src.model.dag_model import AddOp, ConcatOp, DagModel, JumpOp

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

# Model 0 shortcut add:
# Shape: 2 -> 2

Poc0Inshape_s = (2,)
Poc0Model_sa = DagModel(
    nn.Linear(2, 2),
    nn.Linear(2, 2),
    (AddOp(), [-1]), # add FM[1] + FM[2]
)
map["0-add"] = (Poc0Inshape_s, Poc0Model_sa)

# Model 0 shortcut concat:
# Shape: 2 -> 2 -> 4

Poc0Model_sc = DagModel(
    nn.Linear(2, 2),
    nn.Linear(2, 2),
    (ConcatOp(1), [-1, None]), # concat FM[1] + FM[2]
)
map["0-con"] = (Poc0Inshape_s, Poc0Model_sc)

# Model 0 shortcut jump:
# Shape: 2 -> 3 -> 4 -> 2

Poc0Model_sj = DagModel(
    nn.Linear(2, 3),
    nn.Linear(3, 4),
    (JumpOp(), [-1]), # jump to FM[1]
)
map["0-jmp"] = (Poc0Inshape_s, Poc0Model_sj)

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
# Shape: 10 -relu-> 10 -add(-2)-> 10 -> 5

Poc4Inshape = (10,)
Poc4Model_a = DagModel(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    (AddOp(), [-1]), # add FM[2] + FM[3]
    nn.ReLU(),
    nn.Linear(10, 5),
)
map["4"] = (Poc4Inshape, Poc4Model_a)

Poc4Model_j = DagModel(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    (JumpOp(), [-2]), # jump to FM[2]
    nn.Linear(10, 5),
)
map["4-jmp"] = (Poc4Inshape, Poc4Model_j)

# Model 5:
# Shape: 1x10x10 -conv-> 5x8x8 -conv-> 5x8x8 -add-> 5x8x8 -> 320 -> 10

Poc5Inshape = (1, 10, 10)
Poc5Model = DagModel(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 5, 3, 1, 1),
    nn.ReLU(),
    nn.Conv2d(5, 5, 3, 1, 1),
    (AddOp(), [-1]), # add FM[3] + FM[5]
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(320, 10),
)
map["5"] = (Poc5Inshape, Poc5Model)

# Model 6:
#                              -conv1-> 4x8x8
# Shape: 1x10x10 -conv-> 5x8x8                -conc-> 10x8x8 -> 640 -> 10
#                              -conv2-> 6x8x8

Poc6Inshape = (1, 10, 10)
Poc6Model = DagModel(
    nn.Conv2d(1, 5, 3),
    nn.ReLU(),
    nn.Conv2d(5, 4, 3, 1, 1), # conv1
    nn.ReLU(),
    (JumpOp(), [-2]), # jump to FM[2]
    nn.Conv2d(5, 6, 3, 1, 1), # conv2
    nn.ReLU(),
    (ConcatOp(1), [-3, None]), # concat FM[4] + FM[7]
    nn.Conv2d(10, 10, 3, 1, 1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(640, 10),
)
map["6"] = (Poc6Inshape, Poc6Model)
