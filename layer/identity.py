from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class IdentityClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)
    

class IdentityServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, torch.nn.Identity)
        super().__init__(socket, ishape, oshape, layer, device)
    
    def offline(self) -> np.ndarray:
        t = time.time()
        rm = self.protocol.recv_offline()
        self.protocol.send_offline(rm)
        self.stat.time_offline += time.time() - t
        return rm
    
    def online(self) -> torch.Tensor:
        t = time.time()
        xrm = self.protocol.recv_online()
        self.protocol.send_online(xrm)
        self.stat.time_online += time.time() - t
        return xrm
