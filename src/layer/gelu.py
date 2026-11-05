from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
# import numpy as np
import time
import torch
try:  # Pyfhel is optional; only needed when use_he is enabled
    from Pyfhel import Pyfhel
except ImportError:  # pragma: no cover
    Pyfhel = None
class GELUClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, lyr: torch.nn.Module, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)
        self.layer = lyr
    
    def online(self, xm) -> torch.Tensor:
        t0 = time.time()
        data = self.layer(xm)
        t1 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online += t1 - t0
        return data
    
class GELUServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, torch.nn.GELU)
        super().__init__(socket, ishape, oshape, layer, device)