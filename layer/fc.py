from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from Pyfhel import Pyfhel

class FcClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)

class FcServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, torch.nn.Linear)
        super().__init__(socket, ishape, oshape, layer, device)
    
    def offline(self) -> np.ndarray:
        t0 = time.time()
        rm = self.protocol.recv_offline() # recv: r'_i = r_i / m_{i-1}
        t1 = time.time()
        data = self.run_layer_offline(rm) # W_i * r'_i
        t2 = time.time()
        self.protocol.send_offline(data) # send: (W_i * r'_i) .* m_i - s_i
        t3 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline += t3 - t0
        return rm
    
    def online(self) -> torch.Tensor:
        t0 = time.time()
        xmr_i = self.protocol.recv_online() # recv: xmr_i = x_i - r_i / m_{i-1}
        t1 = time.time()
        data = self.layer(xmr_i) # W_i * (x_i - r_i / m_{i-1})
        t2 = time.time()
        self.protocol.send_online(data) # send: (W_i * (x_i - r_i / m_{i-1})) .* m_i + s_i
        t3 = time.time()
        self.stat.time_online_comp += t2 - t1
        self.stat.time_online += t3 - t0
        return xmr_i
