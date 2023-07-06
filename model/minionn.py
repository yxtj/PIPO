import torch
import torch.nn as nn

inshape = (3, 32, 32)

# Shape:
# input (3, 32, 32) conv (64, 32, 32) conv (64, 32, 32) pool (64, 16, 16)
# conv (64, 16, 16) conv (64, 16, 16) pool (64, 8, 8)
# conv (64, 8, 8) conv (64, 8, 8) conv (16, 8, 8) 
# flatten (1024) linear (10)

def build(weight_path=None):
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        # nn.AvgPool2d(2, 2),
        
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        # nn.AvgPool2d(2, 2),
        
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 1, 1, 0),
        nn.ReLU(),
        nn.Conv2d(64, 16, 1, 1, 0),
        nn.ReLU(),
        
        nn.Flatten(),
        nn.Linear(1024, 10),
        #nn.Softmax(dim=1),
    )
    # load pretrained weights
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    return model

