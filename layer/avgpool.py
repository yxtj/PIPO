from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class AvgPoolClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel,
                 layer:torch.nn.AvgPool2d, device:str) -> None:
        super().__init__(socket, ishape, oshape, he, device)
        

class AvgPoolServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, torch.nn.AvgPool2d)
        super().__init__(socket, ishape, oshape, layer, device)
    
    def offline(self) -> np.ndarray:
        t0 = time.time()
        rm = self.protocol.recv_offline() # r'_i = r_i / m_{i-1}
        t1 = time.time()
        data = self.layer(rm) # avg_pool(r'_i)
        t2 = time.time()
        self.protocol.send_offline(data)
        t3 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline += t3 - t0
        return rm
    
    def online(self) -> torch.Tensor:
        t0 = time.time()
        xmr_i = self.protocol.recv_online() # xmr_i = x_i - r_i / m_{i-1}
        t1 = time.time()
        data = self.layer(xmr_i) # avg_pool(x_i - r_i / m_{i-1})
        t2 = time.time()
        self.protocol.send_online(data)
        t3 = time.time()
        self.stat.time_online_comp += t2 - t1
        self.stat.time_online += t3 - t0
        return xmr_i
