from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
# import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class FlattenClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel) -> None:
        super().__init__(socket, ishape, oshape, he)
        self.layer = torch.nn.Flatten()
    
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        data = self.layer(xm)
        self.stat.time_online += time.time() - t
        return data


class FlattenServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, torch.nn.Flatten)
        super().__init__(socket, ishape, oshape, layer)
        
    def setup(self, last_lyr: LocalLayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        # flatten the last m and forward to the next layer
        t = time.time()
        last_pto = last_lyr.protocol if last_lyr is not None else None
        m = self.layer(last_pto.m) if last_pto is not None else None
        self.protocol.setup(self.ishape, self.oshape, s=0, m=m, last=last_pto)
        self.stat.time_offline += time.time() - t
    