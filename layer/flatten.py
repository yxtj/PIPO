from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
# import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class FlattenClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)
        self.layer = torch.nn.Flatten()
    
    def online(self, xm) -> torch.Tensor:
        t0 = time.time()
        data = self.layer(xm)
        t1 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online += t1 - t0
        return data


class FlattenServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, torch.nn.Flatten)
        super().__init__(socket, ishape, oshape, layer, device)
        
    def setup(self, last_lyr: LocalLayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        # flatten the last m and forward to the next layer
        t = time.time()
        last_pto = last_lyr.protocol if last_lyr is not None else None
        self.protool = self.protocol.setup_local(self.ishape, self.oshape, last_pto, 'flatten')
        self.stat.time_offline += time.time() - t
    