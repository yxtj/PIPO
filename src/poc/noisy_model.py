# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from src.model.dag_model import DagModel


class NoisyModel(nn.Module):
    def __init__(self, m: nn.Module, noise: float):
        super().__init__()
        self.m = m
        self.noise = noise

    def gen_noise_factor(self, x):
        return 1 - self.noise + 2*self.noise*torch.rand_like(x)

    def forward(self, x: torch.Tensor):
        if isinstance(self.m, DagModel):
            entries = [e for e in self.m.flat_entries() if not e.is_block]
            fms = {0: x}
            for e in entries:
                inputs = [fms[s] for s in e.sources]
                for i, inp in enumerate(inputs):
                    if isinstance(e.module, (nn.ReLU, nn.MaxPool2d)):
                        inputs[i] = inp * self.gen_noise_factor(inp)
                with torch.no_grad():
                    out = e.module(*inputs) if len(inputs) > 1 else e.module(inputs[0])
                fms[e.dest] = out
            return fms[len(entries)]
        for lyr in self.m:
            if isinstance(lyr, (nn.ReLU, nn.MaxPool2d)):
                x = x * self.gen_noise_factor(x)
            x = lyr(x)
        return x
