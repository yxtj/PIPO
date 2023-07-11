import torch
import torch.nn as nn
from collections import defaultdict

from .shortcut import ShortCut

class SequentialShortcut(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.bindings = defaultdict(list)
        for i, lyr in enumerate(self):
            if isinstance(lyr, ShortCut):
                idx = i + lyr.relOther + 1
                self.bindings[idx].append(i)
        # print(self.bindings)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        d = x
        for i, lyr in enumerate(self):
            if i in self.bindings:
                for j in self.bindings[i]:
                    self[j].update(d)
            d = lyr(d)
        return d
