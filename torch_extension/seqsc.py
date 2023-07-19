import torch
import torch.nn as nn
from collections import defaultdict

from .shortcut import ShortCut


class SequentialShortcut(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.shortcuts = {} # {shortcut_layer_idx: required_layer_idx}
        self.dependency = defaultdict(list) # {intermediate_result_idx: [shortcut_layer_idx]}
        for i, lyr in enumerate(self):
            if isinstance(lyr, ShortCut):
                for j in lyr.relOther:
                    idx = i + j
                    self.shortcuts[i] = idx
                    self.dependency[idx + 1].append(i)
        # print(self.dependency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x
        for i, lyr in enumerate(self):
            if i in self.dependency:
                for j in self.dependency[i]:
                    self[j].update(i-j-1, d)
            d = lyr(d)
        return d

    def get_shortcut(self) -> dict[int, int]:
        '''return {shortcut_layer_idx: required_layer_idx}'''
        return self.shortcuts

    def get_dependency(self) -> dict[int, list[int]]:
        '''return {intermediate_result_idx: [shortcut_layer_idx]}
        The input is 0-th intermediate result. The output of i-th layer is (i+1)-th intermediate result.
        '''
        return self.dependency
