from .base import LocalLayerClient, LocalLayerServer

from socket import socket
from typing import Union
# import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class ReLUClient(LocalLayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)
        self.layer = torch.nn.ReLU()
    
    def online(self, xm) -> torch.Tensor:
        t = time.time()
        data = self.layer(xm)
        self.stat.time_online += time.time() - t
        return data
    
class ReLUServer(LocalLayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, torch.nn.ReLU)
        super().__init__(socket, ishape, oshape, layer, device)
        