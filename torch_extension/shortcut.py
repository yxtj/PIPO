import torch
import torch.nn as nn

class ShortCut(nn.Module):
    '''
    An abstract shortcut layer that connects the output of the previous layer
     with the output of another layer.
    @param relOther: The relateive index of the other layer to connect to. (-1 is the last layer)
    '''
    def __init__(self, relOther:int) -> None:
        super().__init__()
        assert relOther < 0
        self.relOther = relOther
        self.y = 0
    
    def forward2(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        '''
        The actual forward function. 
        "x" is the output of ealiest layer. "y" is the output of the previous layer.
        '''
        raise NotImplementedError

    def update(self, y:torch.Tensor) -> None:
        self.y = y

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # put the early layer's data (y) before the later one (x)
        return self.forward2(self.y, x)

# %% addition layer

class Addition(ShortCut):
    def __init__(self, relOther:int) -> None:
        if relOther == -1:
            print('Warning: Addition layer is connected to the last layer.')
        super().__init__(relOther)
    
    def forward2(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return x + y

# %% concatenation layer

class Concatenation(ShortCut):
    def __init__(self, relOther:int, dim:int=1) -> None:
        super().__init__(relOther)
        self.dim = dim
    
    def forward2(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        return torch.cat((x, y), dim=self.dim)

