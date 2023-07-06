import torch
import torch.nn as nn

class ShortCut(nn.Module):
    def __init__(self, otherlayer:int) -> None:
        '''
        Add the output of the previous layer to the output of another layer.
        otherlayer: the relative index of the other layer in the model
        '''
        super().__init__()
        self.otherlayer = otherlayer
        
    def forward(self, x, y=None):
        if y is None:
            return x
        return x+y
    