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
        t0 = time.time()
        rm = self.protocol.recv_offline()
        # t1 = time.time()
        self.protocol.send_offline(rm)
        t2 = time.time()
        self.stat.time_offline += t2 - t0
        return rm
    
    def online(self) -> torch.Tensor:
        t0 = time.time()
        xrm = self.protocol.recv_online()
        t1 = time.time()
        self.protocol.send_online(xrm)
        t2 = time.time()
        # self.stat.time_online_recv += t1 - t0
        # self.stat.time_online_send += t2 - t1
        self.stat.time_online += t2 - t0
        return xrm
