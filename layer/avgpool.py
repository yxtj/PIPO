from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class AvgPoolClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel,
                 layer:torch.nn.AvgPool2d) -> None:
        super().__init__(socket, ishape, oshape, he)
        

class AvgPoolServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module) -> None:
        assert isinstance(layer, torch.nn.AvgPool2d)
        super().__init__(socket, ishape, oshape, layer)
    
    def offline(self) -> np.ndarray:
        t = time.time()
        rm = self.protocol.recv_offline() # r'_i = r_i / m_{i-1}
        data = self.layer(rm) # avg_pool(r'_i)
        self.protocol.send_offline(data)
        self.stat.time_offline += time.time() - t
        return rm
    
    def online(self) -> torch.Tensor:
        t = time.time()
        xmr_i = self.protocol.recv_online() # xmr_i = x_i - r_i / m_{i-1}
        data = self.layer(xmr_i) # avg_pool(x_i - r_i / m_{i-1})
        self.protocol.send_online(data)
        self.stat.time_online += time.time() - t
        return xmr_i
