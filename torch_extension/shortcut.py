from typing import List, Union
from collections import OrderedDict
import torch
import torch.nn as nn

__ALL__ = ['ShortCut', 'Jump', 'Addition', 'Concatenation']

class ShortCut(nn.Module):
    '''
    An abstract shortcut layer that connects the output of the previous layer
     with the output of another layer.
    @param relOther: The relateive index(indices) of the other layer to connect to. (-1 is the last layer)
    '''
    def __init__(self, relOther:Union[int, List[int]]) -> None:
        super().__init__()
        if isinstance(relOther, int):
            relOther = [relOther]
        assert all(e<0 for e in relOther)
        self.relOther = relOther
        self.buffer = OrderedDict((e,None) for e in relOther)

    def update(self, relIdx:int, y:torch.Tensor) -> None:
        self.buffer[relIdx] = y

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# %% jump layer

class Jump(ShortCut):
    '''
    Copy the output of an early layer. (Make a flywire connection)
    '''
    def __init__(self, relOther:int) -> None:
        assert isinstance(relOther, int) and relOther != -1, "It is trivial to connect to the last layer."
        super().__init__(relOther)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.buffer[self.relOther[0]]

    def __repr__(self) -> str:
        return f'Jump({self.relOther})'

# %% addition layer

class Addition(ShortCut):
    '''
    Add the last layer with others together.
    The last layer (index -1) is automatically included, and need not be specified in 'relOther'.
    '''
    def __init__(self, relOther:Union[int, List[int]]) -> None:
        super().__init__(relOther)
        if any(e == -1 for e in self.relOther):
            print('Warning: Addition layer includes the last layer more than once.')
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        self.buffer[-1] = x
        return torch.stack(tuple(self.buffer.values())).sum(dim=0)

    def __repr__(self) -> str:
        return f'Addition({self.relOther})'

# %% concatenation layer

class Concatenation(ShortCut):
    '''
    Concatenate the last layer with others together.
    The last layer (index -1) is automatically included, and need not be specified in 'relOther'.
    @param dim: The dimension to concatenate.
    @param order: The order of the layers to concatenate. (default: None, which means the order is ('relOther', -1))
    '''
    def __init__(self, relOther:Union[int, List[int]], dim:int=1, order:List[int]=None) -> None:
        super().__init__(relOther)
        self.dim = dim
        if any(e == -1 for e in self.relOther):
            print('Warning: Concatenation layer includes the last layer more than once.')
        if order is None:
            order = self.relOther + [-1]
        else:
            assert len(order) == len(self.relOther) + 1
            assert all(a==b for a, b in zip(sorted(order), sorted(self.relOther + [-1])))
        self.order = order
        self.buffer = OrderedDict((e,None) for e in self.order)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        self.buffer[-1] = x
        return torch.cat(tuple(self.buffer.values()), dim=self.dim)

    def __repr__(self) -> str:
        return f'Concatenation({self.relOther}, dim={self.dim}, order={self.order})'
