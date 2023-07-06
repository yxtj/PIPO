import torch
import torch.nn as nn
from .shortcut import ShortCut

class SequentialBuffer(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        # TODO: bind buffer to addition layer. Make addition layer work with forward(x)
        self.to_buffer = []
        for i, lyr in enumerate(self):
            if isinstance(lyr, ShortCut):
                idx = i + lyr.otherlayer
                self.to_buffer.append(idx)

    def forward(self, input):
        d = input
        buffer = { -1: input }
        for i, lyr in enumerate(self):
            if isinstance(lyr, ShortCut):
                d = lyr(d, buffer[i + lyr.otherlayer])
            else:
                d = lyr(d)
            if i in self.to_buffer:
                buffer[i] = d
        return d
